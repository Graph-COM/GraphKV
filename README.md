# Graph_KV

## Overview

This is the official implementation of paper 'Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models', Haoyu Wang, Peihao Wang, Mufei Li, Shikun Liu, Siqi Miao, Zhangyang Wang, Pan Li.

## STEP 0.1 Environment Setup

To setup the environment, follow the scripts:

```
conda create -n graphkv python==3.10.16
conda activate graphkv
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.50.0
pip install accelerate
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install pandas

# Below is to set up server to serve LLMs, (Following the Block-RAG paper)
pip install fires
pip install flask_cors
```

## STEP 0.2 Tuned Model Parameters

Below are the tuned-model prameters adopted in Graph-KV.

| Item                                   | Repository |
| - | - |
| 8B-Block | [🤗 ldsjmdy/Tulu3-Block-FT](https://huggingface.co/ldsjmdy/Tulu3-Block-FT) |
| 8B-SFT | [🤗 ldsjmdy/Tulu3-SFT](https://huggingface.co/ldsjmdy/Tulu3-SFT) |
| 8B-RAG | [🤗 ldsjmdy/Tulu3-RAG](https://huggingface.co/ldsjmdy/Tulu3-RAG) |

## Step 0.3 Dataset Processing

### To Prepare for 2Wiki, NarritiveQA, Trivia QA, HotpotQA from Scratch
We follow [Block-Attention for Efficient Prefilling](https://arxiv.org/abs/2409.15355) to pre-process the data for the [2Wiki](https://arxiv.org/abs/2011.01060), [NarritiveQA](https://arxiv.org/abs/1712.07040), [Trivia QA](https://arxiv.org/abs/1705.03551), [HotpotQA](https://arxiv.org/abs/1809.09600).

For obtaining the raw data and pre-processing the data from scratch, please refer to the [original implementation](https://github.com/TemporaryLoRA/Block-Attention). 

Instructions for directly downloading the processed data can be found below.


### To Prepare for Multihop-RAG and MorehopQA from Scratch

We follow the original implementation in [MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/abs/2401.15391) to pre-process the data for Multihop-RAG, and the original implementation in [MoreHopQA: More Than Multi-hop Reasoning](https://arxiv.org/abs/2406.13397) to pre-process the data for Morehop-QA.

Please refer to their repositories [Multihp-RAG](https://github.com/yixuantt/MultiHop-RAG), [MorehopQA](https://github.com/Alab-NII/morehopqa) for dataset pre-processing from scratch.

Instructions for directly downloading the processed data can be found below.


### Download Pre-Processed Datasets

[Huggingface Dataset Link for Processed Data](https://huggingface.co/datasets/Graph-COM/GraphKV)

One may dowload from the huggingface repository listed above to directly download all the required pre-processed data for both RAG (rag) and the Arxiv-QA (arxiv) tasks.

Please refer to the [ReadMe file](https://huggingface.co/datasets/Graph-COM/GraphKV/blob/main/README.md) in the Huggingface Repo for details.

## RAG Task

### Set up inference server

```
CUDA_VISIBLE_DEVICES=0 python3 server/generate_server.py --model ldsjmdy/Tulu3-Block-FT --port 8771 --dtype bfloat16
```

| Component                           | Description                                                                              |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| `CUDA_VISIBLE_DEVICES=0`            | Specifies to use GPU device 0. Modify this if using multiple GPUs.                       |
| `--model`                           | The Hugging Face model to load. Options:                                                 |
|                                     |     `ldsjmdy/Tulu3-Block-FT` – Block-level fine-tuned model                              |
|                                     |     `ldsjmdy/Tulu3-RAG` – RAG-tuned model                                                |
|                                     |     `ldsjmdy/Tulu3-SFT` – Supervised fine-tuned model                                    |
| `--port 8771`                       | Port on which the server will listen. Change this if the port is in use.                 |


### Inference

```
python inference.py --pcw vanilla --model ldsjmdy/Tulu3-Block-FT --task nqa --port 8771
python inference.py --pcw gapemp --model ldsjmdy/Tulu3-Block-FT --task nqa --port 8771
python inference.py --pcw block --model ldsjmdy/Tulu3-Block-FT --task nqa --port 8771
python inference.py --pcw gapemp_appr --model ldsjmdy/Tulu3-Block-FT --task nqa --port 8771 --top_k 5
```

| Argument             | Description                                                                |
| -------------------- | -------------------------------------------------------------------------- |
| `--pcw`              | Method for parallel context window (PCW) attention implementation. Options:  |
|                      |     `vanilla` – Sequential encoding without PCW                                |
|                      |     `gapemp` – Uses  Graph-KV                           |
|                      |     `block` – Uses block-RAG                             |
|                      |     `gapemp_appr` – Approximate version of Grpah-KV                 |
| `--model`            | Model name loaded by the server (must match the one started on the server) |
| `--task`             | Task type. Currently support  'nqa, 2wiki, tqa, hqa, morehopqa, multihopqa'                    |
| `--port`             | Port number to communicate with the generation server                      |
| `--top_k` (optional) | **Only used in `gapemp_appr`** mode. Specifies top-K passages to select    |
|                      | based on relevance scores during information retrieval.                      |


**Note** that the pre-computed results of each method is also provided, in the [results](https://huggingface.co/datasets/Graph-COM/GraphKV/tree/main/results) folder of the Huggingface repository.

### Evaluation

```
python rag_eval.py --input [PATH]
```
| Argument  | Description                                                             |
| --------- | ----------------------------------------------------------------------- |
| `--input` | Path to the `.jsonl` file containing inference outputs to evaluate.     |
|           | For example: `./results/tqa/ldsjmdyTulu3-Block-FT/vanilla_ascent.jsonl` |



## Arxiv-QA Task

The data can be found in [(arxiv) folder](https://huggingface.co/datasets/Graph-COM/GraphKV/tree/main/datahub/arxiv) of the Huggingface Repository.

### Inference

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python arxiv_inference.py --pcw gapemp_graph --batch_size 1 --order first --model ldsjmdy/Tulu3-Block-FT
```
```
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29501 arxiv_inference.py --pcw vanilla --batch_size 1 --order first --model ldsjmdy/Tulu3-Block-FT
```


| Argument                         | Description                                                             |
| -------------------------------- | ----------------------------------------------------------------------- |
| `CUDA_VISIBLE_DEVICES=0,1,2,3`   | Specifies the GPUs to use for inference. This example uses 4 GPUs.      |
| `python arxiv_inference.py`      | The evaluation script for Arxiv-QA.                                      |
| `--pcw gapemp_graph`             | The attention method   Example: `gapemp_graph` , `vanilla`, `block`    |
| `--batch_size 1`                 | Number of distractors. 1 means no distractors.         |
| `--order first`                  | Put the related paper group  at `last` or `first`. |
| `--model ldsjmdy/Tulu3-Block-FT` | Specifies the model to be used for inference.                           |


**Note** that the pre-computed results of each method is also provided, in the [results](https://huggingface.co/datasets/Graph-COM/GraphKV/tree/main/results) folder of the Huggingface repository.

### Evaluation

```
python arxiv_eval.py --model ldsjmdy/Tulu3-Block-FT --pcw gapemp_graph --batch_size 1 --order last
```

| Argument       | Description                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| `--model`      | Name of the model to evaluate. Example: `ldsjmdy/Tulu3-Block-FT`.                                        |
| `--pcw`        | The attention method                                   |
|                | Example: `gapemp_graph` , `vanilla`, `block`                                    |
| `--batch_size` | number of distractors. 1: no distractors. could be [1,2,3]      |
| `--order`      | Direct related paper ordering strategy. `last` means putting in the last. 'first' means putting in the first |




# Citation

If you find this work or repository helpful, please consider citing:

```bibtex
@article{wang2025graph,
  title={Graph-KV: Breaking Sequence via Injecting Structural Biases into Large Language Models},
  author={Wang, Haoyu and Wang, Peihao and Li, Mufei and Liu, Shikun and Miao, Siqi and Wang, Zhangyang and Li, Pan},
  journal={arXiv preprint arXiv:2506.07334},
  year={2025}
}