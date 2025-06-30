import json
import fire
import torch
from flask_cors import CORS
from flask import Flask, request
import argparse
import os

from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict, Union

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaConfig, LlamaForCausalLM
from transformers import (
    AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, GenerationConfig, AutoConfig
)


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pcw import gapemp, gapemp_appr
from server.block_generate_server import block_generate




app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/generate_block', methods=['POST'])
def _block_generate():
    form = request.get_json()
    del form["blocks"][1] # To remove middle
    generated = block_generate(
        blocks=form["blocks"][:-1],
        instruction=form["blocks"][-1],
        generation_config=generation_config,
        model=model,
        emb=emb,
        tokenizer=tokenizer,
        num_local_attention_blocks=form.get("num_local_attention_blocks", 10000),
    )
    print("generated: ", generated)
    return {"ret": 0, "generated": generated, "message": ""}

@app.route('/generate_vanilla', methods=['POST'])
def _sequential_generate():
    form = request.get_json()
    # merge the blocks together
    all_input = ''.join(form["blocks"])
    input_ids = tokenizer(all_input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
    context_length = input_ids.size(-1)
    # infer with model
    response = model.generate(
        input_ids=input_ids.to(model.device),
        max_new_tokens=1024,
        num_beams=1,
        do_sample=False,
        temperature=1.0,)[0]
    generated = tokenizer.decode(response[context_length:], skip_special_tokens=True)
    print("generated: ", generated)
    return {"ret": 0, "generated": generated, "message": ""}

@app.route('/generate_gapemp', methods=['POST'])
def _gapemp_generate():
    form = request.get_json()
    prefix = form['blocks'][0]
    middle = form['blocks'][1]
    contexts = form['blocks'][2:-1]
    query = form['blocks'][-1]
    generated = gapemp(tokenizer, model, emb, prefix, middle,query, contexts, args.model, 1, 1, None)
    print("generated: ", generated)
    return {"ret": 0, "generated": generated, "message": ""}

@app.route('/generate_gapemp_appr', methods=['POST'])
def _gapemp_appr_generate():
    form = request.get_json()
    prefix = form['blocks'][0]
    middle = form['blocks'][1]
    contexts = form['blocks'][2:-1]
    query = form['blocks'][-1]
    top_k = form['top_k']
    generated = gapemp_appr(tokenizer, model, emb, prefix, middle,query, contexts, args.model, 1, 1, top_k)
    print("generated: ", generated)
    return {"ret": 0, "generated": generated, "message": ""}

@dataclass
class Args:
    model: str
    port: int
    dtype: str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ldsjmdy/Tulu3-Block-FT', choices=['ldsjmdy/Tulu3-Block-FT', 'ldsjmdy/Tulu3-RAG', 'ldsjmdy/Tulu3-SFT'])
    parser.add_argument('--port', type = int, default = 8765)
    parser.add_argument('--dtype', type = str, default = '')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token # essential for padding

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
        #attn_implementation="eager"
    )
    model.eval()
    config: LlamaConfig = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.model)
    emb: LlamaRotaryEmbedding = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)
    emb.eval()

    generation_config = GenerationConfig(
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.0,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
        stop_strings=['<|im_end|>', "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "</s>", "Question:"]
    )
    app.run(host="0.0.0.0", port=args.port)
