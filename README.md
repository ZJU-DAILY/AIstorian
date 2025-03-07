# AIstorian

This repo contains code for "AIstorian lets AI be a historian: A KG-powered multi-agent system for accurate biography generation" Please see our paper `AIstorian-Full-Version.pdf` for technique details.


## Installation

1. Install evaluate for metric:

    ```python
    git clone https://github.com/huggingface/evaluate.git
    ```

2. Install Qwen-Agent:

    ```python
    git clone https://github.com/huggingface/evaluate.git
    ```

3. Install LLM training framework: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)


## Datasets

Jingshi: Hundrads classical Chinese Jingshi(advanced scholars) Biography data during Qing Dynstay, spanning multiple dimensions of information such as their life experiences, official careers, and family backgrounds. It offers a wealth of primary - hand materials for delving into numerous domains like the Qing Dynasty's imperial examination system, historical and cultural studies, and social transformations.

This data set is exclusively for non-commercial academic research. Scholars or organizations intending to utilize it must first complete this application form and send it to the designated email address. When submitting the application, it is imperative to list or attach 1-2 representative publications in the field of classical Chinese that you (or your team) have produced in the past six years, in order to demonstrate your research credentials in the relevant areas. Upon receipt and approval of your application, we will provide you with the download link and the decompression password for the data set. All users are required to strictly abide by all the stipulated usage conditions; non-compliance will result in the revocation of the authorization. It is also noteworthy that our data set will be open-sourced simultaneously with the publication of the relevant books, with the aim of promoting the exchange and advancement of academic research, and further driving in-depth exploration and dissemination of knowledge regarding Qing Dynasty Jinshi and related historical and cultural subjects.


## Running

### Index contribution


### Two-step model training

1. Download basemodel: [Qwen](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

2. Prepare your data following the requirement of LLaMA-Factory

3. Run the SFT

4. Merge lora to basemodel

5. Run the SimPO

### Error-aware Generation with Multi-agents

1. start a vllm server of Qwen for Qwen-Agent or use api directly

2. Run the script:

    ```python
    python summary/inference.py \
    --file_path 'test' \
    --model_path '{model_path}' \
    --lora_path '{lora_path}' \
    --save_dir '{save_dir}' \
    --use_index \
    --hallc_infer
    ```