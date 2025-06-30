import os
import json
import argparse
import pandas as pd
import regex
import string
import statistics
import numpy as np

from torch.ao.quantization.fx.utils import all_node_args_except_first
from tqdm import tqdm
from dataclasses import dataclass

from typing import Any, Dict, List, TypedDict


Document = TypedDict("Document", {"title": str, "text": str, "score": float})

SFTDataInstanceInputs = TypedDict("SFTDataInstanceInputs", {
    "input_ids": List[int],
    "labels": List[int]
})

SFTDataInstance = TypedDict("SFTDataInstance", {
    "prompt": str,
    "question": str,
    "answers": List[str],
    "generated": str,
    "inputs": SFTDataInstanceInputs,
    "documents": List[Document]
})


def load_jsonline(fp: str) -> List[Any]:
    with open(fp, "r", encoding="utf-8") as f:
        return [json.loads(i) for i in f]


@dataclass
class EvalArgs:
    input: str



def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


METRICS = [(best_subspan_em, "best_subspan_em"),]

def get_metrics_for_example(example: SFTDataInstance):
    gold_answers = example["answers"]
    model_answer = example["generated"].split("<|im_end|>")[0].split("<|eot_id|>")[0]
    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return example_metrics, example

def eval_old(all_examples):
    all_example_metrics = []
    for example in tqdm(all_examples, total=len(all_examples), desc="Eval: "):
        all_example_metrics.append(get_metrics_for_example(example=example))

    print("All Examples: ", len(all_examples))
    acc_list = []
    for _, metric in METRICS:
        average = statistics.mean(em[metric]  for em, _ in all_example_metrics)
        print(f"{metric}: {average}")
        acc_list.append(average)
    return acc_list


def main():
    parser = argparse.ArgumentParser(description="Run model inference with different modes.")
    parser.add_argument('--model', type=str, default='ldsjmdy/Tulu3-Block-FT', choices=['ldsjmdy/Tulu3-Block-FT', 'ldsjmdy/Tulu3-SFT', 'ldsjmdy/Tulu3-RAG'])
    parser.add_argument('--pcw', type = str, default = 'gapemp_graph', choices = ['gapemp_graph', 'vanilla', 'block'])
    parser.add_argument('--task', type = str, default = 'arxiv', choices = ['arxiv'])
    parser.add_argument('--batch_size', type = int, default = 1, choices = [1,2,3])
    parser.add_argument('--order', type = str, default = 'last', choices = ['first', 'last'])
    parser.add_argument('--seed', type = int, default =42)
    args = parser.parse_args()
    
    if args.batch_size == 1:
        seed_list =  [42]
    else:
        seed_list = [42,43,44]
        
    acc_list = []
    for seed in seed_list:
        output_folder = f'./results/{args.task}/{args.model.replace("/","")}/'
        output_file = output_folder + f'{args.pcw}_{args.batch_size}_{args.order}_{args.seed}.jsonl'
        all_examples: List[SFTDataInstance] = load_jsonline(output_file)
    
        for example_id, example in enumerate(all_examples):
            if isinstance(example['answers'], list):
                continue
            elif isinstance(example['answers'], str):
                all_examples[example_id]['answers'] = [example['answers']]
    
        acc = eval_old(all_examples)[0]
        acc_list.append(acc)
    print(np.mean(acc_list))
    print(np.std(acc_list))
    




if __name__ == '__main__':
    main()