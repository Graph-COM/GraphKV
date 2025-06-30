
import argparse
import numpy as np
import random
import json
from collections import defaultdict
from tqdm import tqdm
import re
import os
import requests
import glob


import torch
from transformers.cache_utils import DynamicCache
from transformers.utils import default_cache_path
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaForCausalLM
from transformers import (
    AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
)


#from torch_geometric.utils import k_hop_subgraph


from utils import seed_everything, build_prefix, build_suffix, docs2blocks, find_hash_id

from pcw import vanilla
from pcw_parallel import gapemp_graph, block, gapemp_graph_batch, block_batch


def extract_number(path):
    filename = os.path.basename(path)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else -1

def load_data(set_id):
    text_folder = "./datahub/arxiv/text_data/"
    question_folder = "./datahub/arxiv/questions/"
    answer_folder = "./datahub/arxiv/answers/"
    text_path = text_folder + f"{set_id}.json"
    question_paths = sorted(glob.glob(question_folder +f'{set_id}.txt'), key = extract_number)
    answer_paths = sorted(glob.glob(answer_folder +f'{set_id}.txt'), key = extract_number)
    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        questions = []
        answers = []
        for question_path in question_paths:
            with open(question_path, 'r', encoding='utf-8') as f:
                question = f.read() 
            questions.append(question)
        for answer_path in answer_paths:
            with open(answer_path, 'r', encoding='utf-8') as f:
                answer = f.read() 
            answers.append(answer)
        return data, questions, answers
    return None, None, None

def get_center_neighbor_from_dict(data_dict):
    value_dict = list(data_dict.values())[0]
    center_node = value_dict["0"]
    neighbor_nodes = []
    for key in value_dict["1"].keys():
        neighbor = value_dict["1"][key]
        if neighbor is not None:
            neighbor_nodes.append(neighbor)
    return center_node, neighbor_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ldsjmdy/Tulu3-Block-FT', choices=['ldsjmdy/Tulu3-Block-FT', 'ldsjmdy/Tulu3-SFT', 'ldsjmdy/Tulu3-RAG'])
    parser.add_argument('--pcw', type = str, default = 'gapemp_star', choices = ['gapemp_graph', 'vanilla', 'block'])
    parser.add_argument('--task', type = str, default = 'arxiv', choices = ['arxiv'])
    parser.add_argument('--batch_size', type = int, default = 1, choices = [1,2,3])
    parser.add_argument('--order', type = str, default = 'last', choices = ['first', 'last'])
    parser.add_argument('--seed', type = int, default =42)
    args = parser.parse_args()

    
    # init model
    
    # for our server or remote server, different cache location:
    if default_cache_path.startswith('/nethome'):
        model_pth = default_cache_path + f"/models--{args.model.replace('/','--')}/snapshots/"
    elif default_cache_path.startswith('/mnt'):
        model_pth = '/mnt/main_storage/peihao/cache' + f"/models--{args.model.replace('/','--')}/snapshots/"
    else:
        raise NotImplementedError
    hash_id = find_hash_id(model_pth)
    model_pth = model_pth + hash_id
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_pth,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token # essential for padding

    model_config = AutoConfig.from_pretrained(model_pth)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_pth,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    model.eval()
    emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=model_config).to(device=model.device, dtype=torch.float32)
    emb.eval()

    # set seed
    seed_everything(int(args.seed))
    set_id_list = list(range(1, 61))
    output_folder = f'./results/{args.task}/{args.model.replace("/","")}/'
    os.makedirs(output_folder, exist_ok=True)
    output_file = output_folder + f'{args.pcw}_{args.batch_size}_{args.order}_{args.seed}.jsonl'
    if args.batch_size == 1:
        for idxidx, set_id in enumerate(tqdm(set_id_list)):
            data_dict, questions, answers = load_data(set_id)
            value_dict = list(data_dict.values())[0]
            center_node = value_dict["0"]
            neighbor_nodes = []
            for key in value_dict["1"].keys():
                neighbor = value_dict["1"][key]
                if neighbor is not None:
                    neighbor_nodes.append(neighbor)
            
            for qid, (question, answer) in enumerate(zip(questions, answers)):
                response_dict = {}
                if 'Tulu' in args.model:
                    prefix = "<|user|>\nYou are an intelligent AI assistant. You will first read the related works of a paper, then you will read the paper. Then answer the question.\n\n"
                    query = f"\n Question: {question} \n<|assistant|>\n"
                
                if args.pcw == 'vanilla':
                    prompt = prefix + '\n\n'.join(neighbor_nodes) + '\n\n Now please read the paper:' + center_node + query
                    generated = vanilla(tokenizer, model, prompt, 1, 1, None)
                elif args.pcw == 'gapemp_graph':
                    generated = gapemp_graph(tokenizer, model, emb, prefix, center_node, neighbor_nodes, query, args.model, 1, 1, None)
                elif args.pcw == 'block':
                    generated = block(tokenizer, model, emb, prefix, center_node, neighbor_nodes, query, args.model, 1, 1, None)
                
                print('generated')
                print(generated)
                print('answer')
                print(answer)
                response_dict['generated'] = generated
                response_dict['answers'] = answer
                if idxidx==0 and qid==0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(json.dumps(response_dict, ensure_ascii=False) + "\n")  
                else:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(response_dict, ensure_ascii=False) + "\n") 
    else:
        for idxidx, set_id in enumerate(tqdm(set_id_list)):
            if args.order in ['last']:
                center_node_list = []
                neighbor_nodes_list = []
                distractor_list = random.sample(set_id_list, args.batch_size-1)
                for distractor_set_id in distractor_list:
                    distractor_dict, _, _ = load_data(distractor_set_id)
                    cn, nn = get_center_neighbor_from_dict(distractor_dict)
                    center_node_list.append(cn)
                    neighbor_nodes_list.append(nn)
                data_dict, questions, answers = load_data(set_id)
                center_node, neighbor_nodes = get_center_neighbor_from_dict(data_dict)
                center_node_list.append(center_node)
                neighbor_nodes_list.append(neighbor_nodes)
            elif args.order == 'first':
                center_node_list = []
                neighbor_nodes_list = []
                data_dict, questions, answers = load_data(set_id)
                center_node, neighbor_nodes = get_center_neighbor_from_dict(data_dict)
                center_node_list.append(center_node)
                neighbor_nodes_list.append(neighbor_nodes)
                distractor_list = random.sample(set_id_list, args.batch_size-1)
                for distractor_set_id in distractor_list:
                    distractor_dict, _, _ = load_data(distractor_set_id)
                    cn, nn = get_center_neighbor_from_dict(distractor_dict)
                    center_node_list.append(cn)
                    neighbor_nodes_list.append(nn)

            
            for qid, (question, answer) in enumerate(zip(questions, answers)):
                response_dict = {}
                if 'Tulu' in args.model:
                    prefix = "<|user|>\nYou are an intelligent AI assistant. You will first read the related works of a paper, then you will read the paper. Then answer the question.\n\n"
                    query = f"\n Question: {question} \n<|assistant|>\n"
                
                if args.pcw == 'vanilla':
                    prompt = prefix
                    if args.order == 'cq':
                        for nn_list in neighbor_nodes_list:
                            prompt += '\n\n'.join(nn_list)
                        prompt+= '\n\n Now please read the paper:'
                        for cn in center_node_list:
                            prompt += cn
                    else:
                        for cn, nn_list in zip(center_node_list, neighbor_nodes_list):
                            prompt += '\n\n'.join(nn_list) + '\n\n Now please read the paper:' + cn
                    prompt += query
                    generated = vanilla(tokenizer, model, prompt, 1, 1, None)
                elif args.pcw == 'gapemp_graph':
                    generated = gapemp_graph_batch(tokenizer, model, emb, prefix, center_node_list, neighbor_nodes_list, query, args.model, 1, 1, None)
                elif args.pcw == 'block':
                    generated = block_batch(tokenizer, model, emb, prefix, center_node_list, neighbor_nodes_list, query, args.model, 1, 1, None)
                
                print('generated')
                print(generated)
                print('answer')
                print(answer)
                response_dict['generated'] = generated
                response_dict['answers'] = answer
                if idxidx==0 and qid==0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(json.dumps(response_dict, ensure_ascii=False) + "\n")  
                else:
                    with open(output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(response_dict, ensure_ascii=False) + "\n") 



if __name__ == '__main__':
    main()