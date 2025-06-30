


import torch


from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers import AutoConfig, GenerationConfig


# tool functions to calculate RoPE

def rotate_half(x):
    # tool function in apply_rotart_pos_emb
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    ## apply RoPE to K
    # the code is from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    # the q_embed is commented, and return onlt K pos, for KV cache calculation
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    #q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    #eturn q_embed, k_embed
    return k_embed.to(dtype=torch.bfloat16)


def apply_pkv_rerotary_position_embeddings(pkv: DynamicCache, emb: LlamaRotaryEmbedding, position_ids = None) -> DynamicCache:
    # this algorithm is to remove the RoPE on Ks (rotate them to start from position 0) 
    # input:   pkv: KV cache with PE
    #          position_ids: the original positions used to encode the PE
    device = pkv.key_cache[0].device
    emb.to(device=device)
    if position_ids is None:  
        # default: start form 0
        position_ids = torch.arange(start=0, end=pkv.key_cache[0].size(-2), dtype=torch.int64, device = device)
    position_ids = position_ids.unsqueeze(dim=0).repeat(repeats=[pkv.key_cache[0].size(0), 1]).to(device = device)
    cos, sin = emb(x=pkv.key_cache[0].to(dtype=torch.float32), position_ids=position_ids)
    for i in range(0, len(pkv.key_cache)):
        new_device = pkv.key_cache[i].device
        if new_device != device:
            emb.to(device=new_device)
            cos = cos.to(device = new_device)
            sin = sin.to(device = new_device)
            position_ids = position_ids.to(device=new_device)
            device = new_device
        pkv.key_cache[i] = apply_rotary_pos_emb(
            k=pkv.key_cache[i].to(dtype=torch.float32), cos=cos, sin=-sin, position_ids=position_ids
        )
    return pkv

def apply_pkv_rotary_position_embeddings(pkv: DynamicCache, emb: LlamaRotaryEmbedding, position_ids = None) -> DynamicCache:
    # this algorithm is to rotate RoPE to the desired positions, note the input pkv must be from position 0!!!
    # input:    pkv: KV cache with PE from 0
    #           position_ids: the target positions to rotate PE to
    device = pkv.key_cache[0].device
    emb.to(device=device)
    if position_ids is None:
        position_ids = torch.arange(start=0, end=pkv.key_cache[0].size(-2), dtype=torch.int64, device=device)
    position_ids = position_ids.unsqueeze(dim=0).repeat(repeats=[pkv.key_cache[0].size(0), 1]).to(device = device)
    cos, sin = emb(x=pkv.key_cache[0].to(dtype=torch.float32), position_ids=position_ids)
    for i in range(0, len(pkv.key_cache)):
        new_device = pkv.key_cache[i].device
        if new_device != device:
            emb.to(device=new_device)
            cos = cos.to(device = new_device)
            sin = sin.to(device = new_device)
            position_ids = position_ids.to(device=new_device)
            device = new_device
        pkv.key_cache[i] = apply_rotary_pos_emb(
            k=pkv.key_cache[i].to(dtype=torch.float32), cos=cos, sin=sin, position_ids=position_ids
        )
    return pkv



# toll functions to cut KV

def cut_pkv(pkv, positions):
    # cut the KV cache, leave only elements on positions
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id][:,:,positions,:]
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id][:,:,positions,:]
    return pkv
    
def flatten_pkv(pkv, mask):
    # flatten the pkv, remove the padding tokens
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id].transpose(1, 2).flatten(0, 1)[mask].unsqueeze(0).transpose(1, 2)
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id].transpose(1, 2).flatten(0, 1)[mask].unsqueeze(0).transpose(1, 2)
    return pkv

def stack_pkv(pkv, bsz):
    # stack the pkc towards batch_size
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id].repeat(bsz, 1, 1, 1)
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id].repeat(bsz, 1, 1, 1)
    return pkv

def concact_pkv(pkv1, pkv2):
    # concact the two pkvs together, merge into pkv2
    for layer_id in range(len(pkv2.key_cache)):
        pkv2.key_cache[layer_id] = torch.concat((pkv1.key_cache[layer_id], pkv2.key_cache[layer_id]), dim = 2)
        pkv2.value_cache[layer_id] = torch.concat((pkv1.value_cache[layer_id], pkv2.value_cache[layer_id]), dim = 2)
    return pkv2

def topk_pkv(pkv, top_k):
    # concact the two pkvs together, merge into pkv2
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id][-top_k:, :, :, :]
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id][-top_k:, :, :, :]
    return pkv


# pcw functions

def vanilla(tokenizer, model, prompt, temperature = 1, scale =1, mode = None):
    with torch.no_grad():
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        response = model.generate(
                    input_ids=input_ids.to(model.device), 
                    use_cache=True, 
                    eos_token_id=[tokenizer.eos_token_id], 
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    )[0]
        response = tokenizer.decode(token_ids=response[input_ids.shape[-1]:].tolist(), skip_special_tokens=False)
        return response

def gapemp(tokenizer, model, emb, prefix, middle, query, contexts, model_name, temperature, scale, mode):
    # mode: inference or attention
    with torch.no_grad():
        prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        middle_input_ids = tokenizer(middle, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        len_prefix = prefix_input_ids.shape[1]
        len_middle = middle_input_ids.shape[1]
        len_query = query_input_ids.shape[1]
        context_input_ids = tokenizer(contexts, return_tensors='pt', truncation=True, max_length=8192-len_prefix-len_query-len_middle-256, padding=True, add_special_tokens=False).input_ids
        context_mask = (context_input_ids != tokenizer.pad_token_id).reshape(-1)
        bsz = context_input_ids.shape[0]
        len_flatten_contexts = context_mask.sum().item()
    
        # this is to get the context pkv
        context_outputs = model(
            context_input_ids.to(model.device),
            use_cache=True,
        )
        context_pkv = context_outputs.past_key_values
        context_pkv = apply_pkv_rerotary_position_embeddings(context_pkv, emb)
        context_pkv = flatten_pkv(context_pkv, context_mask)
        
        
        context_pkv_clone = type(context_pkv)()
        context_pkv_clone.key_cache = [t.clone().detach() for t in context_pkv.key_cache]
        context_pkv_clone.value_cache = [t.clone().detach() for t in context_pkv.value_cache]

        
        context_pkv = apply_pkv_rotary_position_embeddings(context_pkv, emb)
        context_pkv = stack_pkv(context_pkv, bsz)

        context2_outputs = model(
            context_input_ids.to(model.device),
            past_key_values=context_pkv,
            use_cache=True,
        )

        context_pkv = context2_outputs.past_key_values
        position_of_context2 = torch.arange(start=0, end=context_input_ids.shape[-1], dtype=torch.int64) + len_flatten_contexts
        context_pkv = cut_pkv(context_pkv, position_of_context2)
        context_pkv = apply_pkv_rerotary_position_embeddings(context_pkv, emb, position_of_context2)

        context_pkv = flatten_pkv(context_pkv, context_mask)
        
        context_pkv = concact_pkv(context_pkv_clone, context_pkv)

        prefix_outputs = model(
            prefix_input_ids.to(model.device),
            past_key_values=None,
            use_cache=True,
        )
        prefix_pkv = prefix_outputs.past_key_values
        prefix_pkv = apply_pkv_rerotary_position_embeddings(prefix_pkv, emb)
        
        context_pkv = concact_pkv(prefix_pkv, context_pkv)
        context_pkv = apply_pkv_rotary_position_embeddings(context_pkv, emb)


        generated = query_input_ids
        past_key_values = context_pkv
        num_generate = 0
        answer_ids = generated.to(model.device)
        while num_generate<1024 and generated[0][0]!=tokenizer.eos_token_id:
            outputs = model(generated.to(model.device), 
                            past_key_values = past_key_values,
                            use_cache = True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            generated = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            answer_ids = torch.cat((answer_ids, generated), dim = 1)
            num_generate += 1
        response = tokenizer.decode(token_ids=answer_ids[0][-num_generate:], skip_special_tokens=False)
        return response


def gapemp_appr(tokenizer, model, emb, prefix, middle, query, contexts, model_name, temperature, scale, top_k):
    with torch.no_grad():
        prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        middle_input_ids = tokenizer(middle, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        len_prefix = prefix_input_ids.shape[1]
        len_middle = middle_input_ids.shape[1]
        len_query = query_input_ids.shape[1]
        context_input_ids = tokenizer(contexts, return_tensors='pt', truncation=True, max_length=8192-len_prefix-len_query-len_middle-256, padding=True, add_special_tokens=False).input_ids
        context_mask = (context_input_ids != tokenizer.pad_token_id).reshape(-1)
        topk_context_mask = (context_input_ids[-top_k:,:] != tokenizer.pad_token_id).reshape(-1)
        bsz = context_input_ids.shape[0]
        len_flatten_contexts = context_mask.sum().item()
        len_flatten_topk_contexts = topk_context_mask.sum().item()
    
        # this is to get the context pkv
        context_outputs = model(
            context_input_ids.to(model.device),
            use_cache=True,
        )
        context_pkv = context_outputs.past_key_values
        context_pkv = apply_pkv_rerotary_position_embeddings(context_pkv, emb)
        
        context_pkv_clone = type(context_pkv)()
        context_pkv_clone.key_cache = [t.clone().detach() for t in context_pkv.key_cache]
        context_pkv_clone.value_cache = [t.clone().detach() for t in context_pkv.value_cache]
        context_pkv_clone = flatten_pkv(context_pkv_clone, context_mask)
        
        context_pkv = topk_pkv(context_pkv, top_k)
        context_pkv = flatten_pkv(context_pkv, topk_context_mask)
        
        
        context_pkv = apply_pkv_rotary_position_embeddings(context_pkv, emb)
        context_pkv = stack_pkv(context_pkv, bsz)

        context2_outputs = model(
            context_input_ids.to(model.device),
            past_key_values=context_pkv,
            use_cache=True,
        )
        context_pkv = context2_outputs.past_key_values
        position_of_context2 = torch.arange(start=0, end=context_input_ids.shape[-1], dtype=torch.int64) + len_flatten_topk_contexts
        context_pkv = cut_pkv(context_pkv, position_of_context2)
        context_pkv = apply_pkv_rerotary_position_embeddings(context_pkv, emb, position_of_context2)

        context_pkv = flatten_pkv(context_pkv, context_mask)
        
        context_pkv = concact_pkv(context_pkv_clone, context_pkv)

        prefix_outputs = model(
            prefix_input_ids.to(model.device),
            past_key_values=None,
            use_cache=True,
        )
        prefix_pkv = prefix_outputs.past_key_values
        prefix_pkv = apply_pkv_rerotary_position_embeddings(prefix_pkv, emb)
        
        context_pkv = concact_pkv(prefix_pkv, context_pkv)
        context_pkv = apply_pkv_rotary_position_embeddings(context_pkv, emb)


        generated = query_input_ids
        past_key_values = context_pkv
        num_generate = 0
        answer_ids = generated.to(model.device)
        while num_generate<1024 and generated[0][0]!=tokenizer.eos_token_id:
            outputs = model(generated.to(model.device), 
                            past_key_values = past_key_values,
                            use_cache = True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            generated = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            answer_ids = torch.cat((answer_ids, generated), dim = 1)
            num_generate += 1
        response = tokenizer.decode(token_ids=answer_ids[0][-num_generate:], skip_special_tokens=True)
        return response

