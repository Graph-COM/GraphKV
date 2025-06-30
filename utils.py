
import numpy as np
import random
import os

import torch


from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import TypedDict, List


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def build_prefix(model_name, task):
    if "Tulu" in model_name:
        prompt = '<|user|>\nYou are an intelligent AI assistant. Please answer questions based on the user\'s instructions. Below are some reference documents that may help you in answering the user\'s question.\n\n'
    return prompt

def build_suffix(prompt, model_name, task):
    if "Tulu" in model_name:
        if task in ['2wiki', 'hqa', 'tqa', 'nqa']:
            prompt = f"Please write a high-quality answer for the given question using only the provided search documents (some of which might be irrelevant). \n Question: {prompt} \n<|assistant|>\n"
        elif task in ['multihopqa']:
            prompt = f"Please write a high-quality answer for the given question using only the provided search documents. The answer to the question is a word or entity. If the provided information is insufficient to answer the question, respond 'Insufficient Information'. Please finally give your answer started with: 'The answer is:'. \n Question: {prompt} \n<|assistant|>\n"
        elif task in ['morehopqa']:
            prompt = f"Please write a high-quality answer for the given question using only the provided search documents. (If the answer is a date, format is as follows: YYYY-MM-DD (ISO standard).) After thinking step by step, give your final answer following 'Answer:' \n Question: {prompt} \n<|assistant|>\n"
    return prompt



def docs2blocks(documents, task, order):
    blocks = []
    if task in ['hqa', '2wiki', 'tqa', 'nqa']:
        if order == 'ascent':
            blocks = [f'- Title: ' + doc['title'] + '\n' + doc['text'] + '\n' for i,doc in enumerate(sorted(documents, key=lambda x: x["score"]))]
        elif order == 'descent':
            blocks = [f'- Title: ' + doc['title'] + '\n' + doc['text'] + '\n' for i, doc in enumerate(sorted(documents, key=lambda x: x["score"], reverse=True))]
        elif order == 'middle':
            sorted_docs = sorted(documents, key=lambda x: x["score"], reverse=True)
            result = []
            left_part = []
            right_part = []
            for i, doc in enumerate(sorted_docs):
                if i % 2 == 0:
                    right_part.append(doc)  
                else:
                    left_part.append(doc)  
            result = left_part[::-1] + right_part  
            blocks = [f'- Title: ' + doc['title'] + '\n' + doc['text'] + '\n' for doc in result]
    elif task in ['multihopqa']:
        if order == 'ascent':
            blocks = [doc['text'].replace("[Excerpt from document]\nt", "T") + '\n' for i,doc in enumerate(sorted(documents, key=lambda x: x["score"]))]
        elif order == 'descent':
            blocks = [doc['text'].replace("[Excerpt from document]\nt", "T") + '\n' for i,doc in enumerate(sorted(documents, key=lambda x: x["score"], reverse = True))]
        elif order == 'middle':
            sorted_docs = sorted(documents, key=lambda x: x["score"], reverse=True)
            result = []
            left_part = []
            right_part = []
            for i, doc in enumerate(sorted_docs):
                if i % 2 == 0:
                    right_part.append(doc)  
                else:
                    left_part.append(doc)  
            result = left_part[::-1] + right_part  
            blocks = [doc['text'].replace("[Excerpt from document]\nt", "T") + '\n' for doc in result]
    elif task == 'morehopqa':
        blocks = [doc + '\n' for doc in documents]
        if order == 'middle':
            sorted_docs = blocks[::-1]
            result = []
            left_part = []
            right_part = []
            for i, doc in enumerate(sorted_docs):
                if i % 2 == 0:
                    right_part.append(doc)  
                else:
                    left_part.append(doc)  
            result = left_part[::-1] + right_part  
            blocks = result
    return blocks

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

def find_hash_id(folder_path):
    snapshots_path = folder_path
    hash_ids = os.listdir(snapshots_path)
    if not hash_ids:
        raise FileNotFoundError(f"No hash ids found under {snapshots_path}")
    
    return hash_ids[0]


