
import argparse
import numpy as np
import random
import json
from collections import defaultdict
from tqdm import tqdm
import re
import os
import requests

import torch


from utils import seed_everything, build_prefix, build_suffix, docs2blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ldsjmdy/Tulu3-Block-FT', choices=['ldsjmdy/Tulu3-Block-FT', 'ldsjmdy/Tulu3-RAG', 'ldsjmdy/Tulu3-SFT'])
    parser.add_argument('--port', type=str, default=8765)
    parser.add_argument('--pcw', type = str, default = 'gaped', choices = ['vanilla', 'gapemp', 'block', 'gapemp_appr'])
    parser.add_argument('--order', type = str, default = 'ascent', choices = ['ascent'])
    parser.add_argument('--task', type = str, default = 'hqa', choices = ['hqa', '2wiki', 'tqa', 'multihopqa','nqa', 'morehopqa'])
    parser.add_argument('--top_k', type = int, default = 3, help = 'this is only for gapemp_appr')
    args = parser.parse_args()

    # set seed
    seed_everything(42)

    # load the dataset
    if args.task == 'hqa':
        test_datas = []
        with open('./datahub/rag/hqa_eval/dataset', "r", encoding="utf-8") as f:
            for line in f:
                test_datas.append(json.loads(line))
        # The entire test set includes 7405 instances
        # each has ['prompt', 'question', 'answers', 'generated', 'inputs', 'documents']
        for data_idx, _ in enumerate(test_datas):
            match = re.search(r"title", test_datas[data_idx]['prompt'], flags=re.IGNORECASE)
            test_datas[data_idx]['prefix'] = test_datas[data_idx]['prompt'][:match.start()]

    elif args.task == '2wiki':
        test_datas = []
        with open('./datahub/rag/2wiki_dev/dataset', "r", encoding="utf-8") as f:
            for line in f:
                test_datas.append(json.loads(line))

    elif args.task == 'tqa':
        test_datas = []
        with open('./datahub/rag/tqa_eval/dataset', "r", encoding="utf-8") as f:
            for line in f:
                test_datas.append(json.loads(line))

    elif args.task == 'nqa':
        test_datas = []
        with open('./datahub/rag/nq_eval/dataset', "r", encoding="utf-8") as f:
            for line in f:
                test_datas.append(json.loads(line))

    elif args.task == 'multihopqa':
        with open('./datahub/rag/multihopqa/GIST-large-Embedding-v0_retrieval_test.json', 'r') as file:
            test_datas = json.load(file)
        key_map = {'query': 'question', 'retrieval_list': 'documents', 'answer': 'answers'}
        new_datas = [
            {key_map.get(k, k): v for k, v in d.items()}
            for d in test_datas
        ]
        test_datas = new_datas
    elif args.task == 'morehopqa':
        test_datas = []
        with open('./datahub/morehopqa/with_human_verification_ascend.jsonl', 'r') as file:
            for line in file:
                test_datas.append(json.loads(line.strip()))
        test_datas = sorted(test_datas, key=lambda x: int(x["no_of_hops"]), reverse=True)
        nh = []
        nl = []
        for tt in test_datas:
            nh.append(tt['no_of_hops'])
            nl.append(len(tt['documents']))
            
    # iterate with the eval data, save the results
    output_folder = f'./results/{args.task}/{args.model.replace("/","")}/'
    os.makedirs(output_folder, exist_ok=True)
    if args.task in ['hqa', '2wiki', 'tqa', 'multihopqa', 'nqa', 'morehopqa']:
        if args.pcw not in ['gapemp_appr']:
            output_file = output_folder + f'{args.pcw}_{args.order}.jsonl'
        else:
            output_file = output_folder + f'{args.pcw}_{args.top_k}_{args.order}.jsonl'
    
    for data_idx, data in enumerate(tqdm(test_datas)):
        # process_data to the input of server
        question = data['question'] #+ 'For questions that require judging the truthfulness based on the provided information, please answer Yes or No. For other types of questions, please search for the information before answering.'
        question = build_suffix(question, args.model, args.task)
        prefix = build_prefix(args.model, args.task)
        # middle" unfinished
        middle = ['']
        if args.task in ['hqa', '2wiki', 'tqa', 'multihopqa', 'nqa', 'morehopqa']:
            unprocessed_contexts = data['documents']
            contexts = docs2blocks(unprocessed_contexts, args.task, args.order)

        # block: 0 prefix, 1 middle, 2:-1 contexts, -1 query
        blocks = [prefix] + middle + contexts + [question]
        # query the LLM server for results
        if args.task in ['2wiki', 'hqa', 'tqa', 'multihopqa', 'nqa', 'morehopqa']:
            if args.pcw not in ['gapemp_appr']:
                r_sequential = requests.post(
                    url=f"http://127.0.0.1:{args.port}/generate_{args.pcw}",
                    data=json.dumps({"blocks": blocks}),
                    headers={"Content-Type": "application/json"}
                )
            else:
                r_sequential = requests.post(
                    url=f"http://127.0.0.1:{args.port}/generate_{args.pcw}",
                    data=json.dumps({"blocks": blocks, "top_k": args.top_k}),
                    headers={"Content-Type": "application/json"}
                )
        if r_sequential.status_code == 200:
            try:
                data['generated'] = r_sequential.json()["generated"]
                print(f'generated: {data["generated"]}')
                print(f'answer: {data["answers"]}')
            except (ValueError, KeyError):
                data['generated'] = 'Blank'
        
        
        # save this result:
        if data_idx == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")  
        else:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n") 






if __name__ == '__main__':
    main()