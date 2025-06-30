


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

def divide_pkv(pkv, split_id):
    # divide the KV cache, return pkv before and after the split id
    after_pkv = type(pkv)()
    for layer_id in range(len(pkv.key_cache)):
        after_pkv.key_cache.append(pkv.key_cache[layer_id][:, :, split_id:, :])
        after_pkv.value_cache.append(pkv.value_cache[layer_id][:, :, split_id:, :])
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id][:, :, :split_id, :]
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id][:, :, :split_id, :]
    return pkv, after_pkv
    
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

def concact_pkv_before(pkv1, pkv2):
    # concact the two pkvs together, merge into pkv2
    for layer_id in range(len(pkv2.key_cache)):
        pkv1.key_cache[layer_id] = torch.concat((pkv1.key_cache[layer_id], pkv2.key_cache[layer_id]), dim = 2)
        pkv1.value_cache[layer_id] = torch.concat((pkv1.value_cache[layer_id], pkv2.value_cache[layer_id]), dim = 2)
    return pkv1



def topk_pkv(pkv, top_k):
    # concact the two pkvs together, merge into pkv2
    for layer_id in range(len(pkv.key_cache)):
        pkv.key_cache[layer_id] = pkv.key_cache[layer_id][-top_k:, :, :, :]
        pkv.value_cache[layer_id] = pkv.value_cache[layer_id][-top_k:, :, :, :]
    return pkv

def init_empty_pkv(pkv_example, total_length):
    new_pkv = type(pkv_example)()
    new_pkv.key_cache = []
    new_pkv.value_cache = []
    for layer_id in range(len(pkv_example.key_cache)):
        B, H, _, D = pkv_example.key_cache[layer_id].shape
        new_pkv.key_cache.append(torch.empty(B, H, total_length, D, dtype=pkv_example.key_cache[layer_id].dtype, device=pkv_example.key_cache[layer_id].device))
        new_pkv.value_cache.append(torch.empty(B, H, total_length, D, dtype=pkv_example.value_cache[layer_id].dtype, device=pkv_example.value_cache[layer_id].device))
    return new_pkv


# pcw functions
def gapemp_graph(tokenizer, model, emb, prefix, center_node, neighbor_nodes, query, model_name, temperature, scale, mode):
    # mode: inference or attention
    with torch.no_grad():
        prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        len_prefix = prefix_input_ids.shape[1]
        len_query = query_input_ids.shape[1]
        middle = '\nNow you will read the center paper and answer a related question: \n'
        center_node = middle + center_node
        center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
        len_center = center_input_ids.shape[-1]
        neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
        len_neighbors = neighbors_input_ids.shape[-1]
        len_flatten_neighbors = 0
        neighbor_position_wo_prefix = []

        for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
            neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
            len_flatten_neighbors+=neighbor_input_ids.shape[-1]
            neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
            neighbor_outputs = model(
                neighbor_input_ids.to(model.device),
                use_cache=True,)
            tmp_pkv = neighbor_outputs.past_key_values
            tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
            expected_position = torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64)+len_prefix
            tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
            if neighbor_id==0:
                flatten_pkv = type(tmp_pkv)()
                flatten_pkv.key_cache = [t.clone().detach() for t in tmp_pkv.key_cache]
                flatten_pkv.value_cache = [t.clone().detach() for t in tmp_pkv.value_cache]
            else:
                flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
        neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
        center_outputs = model(
            center_input_ids.to(model.device),
            past_key_values = flatten_pkv,
            cache_position = torch.arange(start=0, end=len_center, dtype=torch.int64, device = model.device)+len_neighbors+len_prefix,
            use_cache=True,
        )

        flatten_pkv = center_outputs.past_key_values
        
        prefix_outputs = model(
            prefix_input_ids.to(model.device),
            past_key_values=None,
            use_cache=True,
        )
        prefix_pkv = prefix_outputs.past_key_values
        flatten_pkv = concact_pkv(prefix_pkv, flatten_pkv)
        generated = query_input_ids
        cache_position = torch.arange(start=0, end=len_query, dtype=torch.int64, device = model.device) + len_prefix + len_neighbors + len_center
        past_key_values = flatten_pkv
        num_generate = 0
        answer_ids = generated.to(model.device)
        while num_generate<256 and generated[0][0]!=tokenizer.eos_token_id:
            outputs = model(generated.to(model.device), 
                            past_key_values = past_key_values,
                            cache_position = cache_position,
                            use_cache = True)
            cache_position = (cache_position.max()+1).reshape(-1)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            generated = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            answer_ids = torch.cat((answer_ids, generated), dim = 1)
            num_generate += 1
        response = tokenizer.decode(token_ids=answer_ids[0][-num_generate:], skip_special_tokens=True)
        return response
    


def gapemp_graph_batch(tokenizer, model, emb, prefix, center_node_list, neighbor_nodes_list, query, model_name, temperature, scale, mode):
    # mode: inference or attention
    with torch.no_grad():
        prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        len_prefix = prefix_input_ids.shape[1]
        len_query = query_input_ids.shape[1]
        middle = '\nNow you will read the center paper and answer a related question: : \n'
        cursor = 0
        for center_node, neighbor_nodes in zip(center_node_list, neighbor_nodes_list):
            center_node = middle + center_node
            center_input_ids = tokenizer(center_node, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
            len_center = center_input_ids.shape[-1]
            neighbors_input_ids = tokenizer(neighbor_nodes, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
            len_neighbors = neighbors_input_ids.shape[-1]
            len_flatten_neighbors = 0
            neighbor_position_wo_prefix = []

            for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
                len_flatten_neighbors+=neighbor_input_ids.shape[-1]
                neighbor_position_wo_prefix.append(torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64))
                neighbor_outputs = model(
                    neighbor_input_ids.to(model.device),
                    use_cache=True,)
                tmp_pkv = neighbor_outputs.past_key_values
                tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                expected_position = torch.arange(start=0, end=neighbor_input_ids.shape[-1], dtype=torch.int64)+len_prefix
                tmp_pkv = apply_pkv_rotary_position_embeddings(tmp_pkv, emb, expected_position)
                if neighbor_id==0:
                    flatten_pkv = type(tmp_pkv)()
                    flatten_pkv.key_cache = [t.clone().detach() for t in tmp_pkv.key_cache]
                    flatten_pkv.value_cache = [t.clone().detach() for t in tmp_pkv.value_cache]
                else:
                    flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
            neighbor_position_wo_prefix = torch.cat(neighbor_position_wo_prefix, dim=0)
            center_outputs = model(
                center_input_ids.to(model.device),
                past_key_values = flatten_pkv,
                cache_position = torch.arange(start=0, end=len_center, dtype=torch.int64, device = model.device)+len_neighbors+len_prefix,
                use_cache=True,
            )
            flatten_pkv = center_outputs.past_key_values
            if cursor == 0:
                overall_pkv = type(flatten_pkv)()
                overall_pkv.key_cache = [t.clone().detach() for t in flatten_pkv.key_cache]
                overall_pkv.value_cache = [t.clone().detach() for t in flatten_pkv.value_cache]
            else:
                overall_pkv = concact_pkv_before(overall_pkv, flatten_pkv)
            cursor = max(cursor, len_neighbors + len_center)

        
        prefix_outputs = model(
            prefix_input_ids.to(model.device),
            past_key_values=None,
            use_cache=True,
        )
        prefix_pkv = prefix_outputs.past_key_values
        overall_pkv = concact_pkv(prefix_pkv, overall_pkv)
        generated = query_input_ids
        past_key_values = overall_pkv
        cache_position = torch.arange(start=0, end=len_query, dtype=torch.int64, device = model.device) + cursor + len_prefix
        num_generate = 0
        answer_ids = generated.to(model.device)
        while num_generate<256 and generated[0][0]!=tokenizer.eos_token_id:
            outputs = model(generated.to(model.device), 
                            past_key_values = past_key_values,
                            cache_position = cache_position,
                            use_cache = True)
            cache_position = (cache_position.max()+1).reshape(-1)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            generated = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            answer_ids = torch.cat((answer_ids, generated), dim = 1)
            num_generate += 1
        response = tokenizer.decode(token_ids=answer_ids[0][-num_generate:], skip_special_tokens=True)
        return response

def block(tokenizer, model, emb, prefix, center_node, neighbor_nodes, query, model_name, temperature, scale, mode):
    # mode: inference or attention
    with torch.no_grad():
        prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        neighbor_nodes.append('\nNow you will read the center paper and answer a related question: \n')
        neighbor_nodes.append(center_node)
        for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
            neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
            neighbor_outputs = model(
                neighbor_input_ids.to(model.device),
                use_cache=True,)
            tmp_pkv = neighbor_outputs.past_key_values
            tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
            if neighbor_id==0:
                flatten_pkv = type(tmp_pkv)()
                flatten_pkv.key_cache = [t.clone().detach() for t in tmp_pkv.key_cache]
                flatten_pkv.value_cache = [t.clone().detach() for t in tmp_pkv.value_cache]
            else:
                flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
        
        prefix_outputs = model(
            prefix_input_ids.to(model.device),
            past_key_values=None,
            use_cache=True,
        )
        prefix_pkv = prefix_outputs.past_key_values
        prefix_pkv = apply_pkv_rerotary_position_embeddings(prefix_pkv, emb)
        flatten_pkv = concact_pkv(prefix_pkv, flatten_pkv)
        flatten_pkv = apply_pkv_rotary_position_embeddings(flatten_pkv, emb)
        generated = query_input_ids
        past_key_values = flatten_pkv
        num_generate = 0
        answer_ids = generated.to(model.device)
        while num_generate<256 and generated[0][0]!=tokenizer.eos_token_id:
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
    
def block_batch(tokenizer, model, emb, prefix, center_node_list, neighbor_nodes_list, query, model_name, temperature, scale, mode):
    # mode: inference or attention
    with torch.no_grad():
        prefix_input_ids = tokenizer(prefix, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        query_input_ids = tokenizer(query, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
        
        flag = 0
        for center_node, neighbor_nodes in zip(center_node_list, neighbor_nodes_list):
            neighbor_nodes.append(center_node)
            for neighbor_id, neighbor_node in enumerate(neighbor_nodes):
                neighbor_input_ids = tokenizer(neighbor_node, return_tensors='pt', truncation=False, padding=True, add_special_tokens=False).input_ids
                neighbor_outputs = model(
                    neighbor_input_ids.to(model.device),
                    use_cache=True,)
                tmp_pkv = neighbor_outputs.past_key_values
                tmp_pkv = apply_pkv_rerotary_position_embeddings(tmp_pkv, emb)
                if neighbor_id==0 and flag==0:
                    flatten_pkv = type(tmp_pkv)()
                    flatten_pkv.key_cache = [t.clone().detach() for t in tmp_pkv.key_cache]
                    flatten_pkv.value_cache = [t.clone().detach() for t in tmp_pkv.value_cache]
                    flag=1
                else:
                    flatten_pkv = concact_pkv_before(flatten_pkv, tmp_pkv)
        
        prefix_outputs = model(
            prefix_input_ids.to(model.device),
            past_key_values=None,
            use_cache=True,
        )
        prefix_pkv = prefix_outputs.past_key_values
        prefix_pkv = apply_pkv_rerotary_position_embeddings(prefix_pkv, emb)
        flatten_pkv = concact_pkv(prefix_pkv, flatten_pkv)
        flatten_pkv = apply_pkv_rotary_position_embeddings(flatten_pkv, emb)
        generated = query_input_ids
        past_key_values = flatten_pkv
        num_generate = 0
        answer_ids = generated.to(model.device)
        while num_generate<256 and generated[0][0]!=tokenizer.eos_token_id:
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
        