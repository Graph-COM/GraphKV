import os
import json
import argparse
import pandas as pd
import regex
import string
import statistics
import re

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

    for _, metric in METRICS:
        average = statistics.mean(em[metric]  for em, _ in all_example_metrics)
        print(f"{metric}: {average}")


def main():
    parser = argparse.ArgumentParser(description="Run model inference with different modes.")
    parser.add_argument('--input', type=str, help="input path of file")
    parser.add_argument('--mode', type=str, choices=['old'], default = 'old', help="Inference mode.")
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        all_examples: List[SFTDataInstance] = load_jsonline(fp=args.input)
    else:
        all_examples: List[SFTDataInstance] = []
        for f_name in os.listdir(args.input):
            fp = os.path.join(args.input, f_name)
            all_examples.extend(load_jsonline(fp=fp))

    if 'multihopqa' not in args.input:
        for example_id, example in enumerate(all_examples):
            if isinstance(example['answers'], list):
                continue
            elif isinstance(example['answers'], str):
                all_examples[example_id]['answers'] = [example['answers']]

    if args.mode == 'old':
        if 'hqa' in args.input:
            #eval_old(all_examples)
            original_data = []
            with open('../src_ape/tasks/hotpot/hotpot_dev_distractor_v1.json', "r", encoding="utf-8") as f:
                for line in f:
                    original_data.append(json.loads(line))
            original_data = original_data[0]
            comparison_list = []
            bridge_list = []
            for data_idx, origin in enumerate(original_data):
                if origin['type'] == 'bridge':
                    bridge_list.append(data_idx)
                elif origin['type'] == 'comparison':
                    comparison_list.append(data_idx)
            
            comparison_exmaples = [all_examples[i] for i in comparison_list]
            bridge_examples = [all_examples[i] for i in bridge_list]
            print(f'HQA comparison {len(comparison_exmaples)}: ')
            if args.mode == 'old':
                eval_old(comparison_exmaples)
            print(f'HQA bridge {len(bridge_examples)}: ')
            if args.mode == 'old':
                eval_old(bridge_examples)
            print(f'HQA all {len(all_examples)}: ')
            if args.mode == 'old':
                eval_old(all_examples)
            
        elif '2wiki' in args.input:
            original_data = []
            df = pd.read_parquet("./datahub/2wiki/dev.parquet")
            
            comparison_idx = df[df["type"] == "comparison"].index.tolist()
            inference_idx = df[df["type"] == "inference"].index.tolist()
            bridge_comparison_idx = df[df["type"] == "bridge_comparison"].index.tolist()
            compositional_idx = df[df["type"] == "compositional"].index.tolist()

            comparison_examples = [all_examples[i] for i in comparison_idx]
            inference_examples = [all_examples[i] for i in inference_idx]
            bridge_comparison_examples = [all_examples[i] for i in bridge_comparison_idx]
            compositional_examples = [all_examples[i] for i in compositional_idx]

            print(f'2wiki comparison {len(comparison_examples)}:')
            if args.mode == 'old':
                eval_old(comparison_examples)
                
            print(f'2wiki inference {len(inference_examples)}:')
            if args.mode == 'old':
                eval_old(inference_examples)

            print(f'2wiki bridge_comparison {len(bridge_comparison_examples)}:')
            if args.mode == 'old':
                eval_old(bridge_comparison_examples)

            print(f'2wiki compositional {len(compositional_examples)}:')
            if args.mode == 'old':
                eval_old(compositional_examples)


            print(f'2wiki all {len(all_examples)}:')
            if args.mode == 'old':
                eval_old(all_examples)
        elif 'multihopqa' in args.input:
            from data_process.rag.multihop_rag.qa_evaluate2 import extract_answer, calculate_metrics
            type_data = {}
            overall_pred_list = []
            overall_gold_list = []
            for d in all_examples:
                model_answer = d['generated']
                if 'The answer' in model_answer:
                    model_answer = extract_answer(model_answer)
                gold = d['answers']
                if gold:
                    question_type = d['question_type']
                    if question_type not in type_data:
                        type_data[question_type] = {'pred_list': [], 'gold_list': []}
                    type_data[question_type]['pred_list'].append(model_answer)
                    type_data[question_type]['gold_list'].append(gold)
                    overall_pred_list.append(model_answer)
                    overall_gold_list.append(gold)
            for question_type, data in type_data.items():
                precision, recall, f1, accuracy = calculate_metrics(data['pred_list'], data['gold_list'])
                print(f"Question Type: {question_type}")
                print(f" Precision: {precision:.2f}")
                print(f" Recall: {recall:.2f}")
                print(f" F1 Score: {f1:.2f}")
                print(f" accuracy: {accuracy:.2f}")
            

            # Calculate overall evaluation metrics
            overall_precision, overall_recall, overall_f1, overall_accuracy = calculate_metrics(overall_pred_list, overall_gold_list)
            print(f"Overall Metrics:")
            print(f" Precision: {overall_precision:.2f}")
            print(f" Recall: {overall_recall:.2f}")
            print(f" F1 Score: {overall_f1:.2f}")
            print(f" Accuracy: {overall_accuracy:.2f}")
            print(f"overall number: {len(overall_pred_list)}")
        elif 'nqa' in args.input:
            for example_idx, _ in enumerate(all_examples):
                all_examples[example_idx]['answers'] = all_examples[example_idx]['answers'][0]
            eval_old(all_examples)
        elif 'morehopqa' in args.input:
            # process the answers
            
            for example_id, _ in enumerate(all_examples):
                if 'generated' in all_examples[example_id]:
                    matches = list(re.finditer(r'answer:\s*(.*)', all_examples[example_id]['generated'], re.IGNORECASE))
                    if matches:
                        result = matches[-1].group(1)
                        all_examples[example_id]['generated'] = result
                else:
                    #sentences = re.split(r'(?<=[.!?])\s+', all_examples[example_id]['generated'].strip())
                    #all_examples[example_id]['generated'] = sentences[-1] if sentences else ''
                    all_examples[example_id]['generated'] = ''
                
            num_hop_list = [example['no_of_hops'] for example in all_examples]
            hop_idxs = [[] for _ in range(5)]
            for idx, val in enumerate(num_hop_list):
                if 1 <= val <= 5:
                    hop_idxs[val - 1].append(idx)
            for hop in range(5):
                print(f'hop {hop+1}')
                this_hop_example = [all_examples[i] for i in hop_idxs[hop]]
                if len(this_hop_example)>0:
                    eval_old(this_hop_example)
            print(f'Overall: {eval_old(all_examples)}')
        else:
            eval_old(all_examples)
    
if __name__ == '__main__':
    main()