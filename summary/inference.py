import os
import sys
import json
import time
import random
import copy
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from retrieval.utils import query
from summary.metrics.rouge import compute_rouge, compute_rag_metric
from summary.utils.hallucination import filter_documents_by_similarity, hallc_check
from summary.utils.model_infer import model_infer
from summary.utils.formatter import format_references, get_summary_input
from summary.hallucination_correction import hallc_correction


BASE_MODEL = os.path.expanduser("~/pretrain/Qwen2.5-7B-Instruct")


# 计时装饰器
timing_results = {}
def timing_decorator_with_storage(timing_dict):
    """
    一个计时装饰器，用于记录函数执行时间。
    :param timing_dict: 字典，用于存储每个函数的执行时间。
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if func.__name__ not in timing_dict:
                timing_dict[func.__name__] = []
            timing_dict[func.__name__].append(elapsed_time)
            return result
        return wrapper
    return decorator


@timing_decorator_with_storage(timing_results)
def retrieval(query_string, index_path):
    if index_path:
        return query.query_origin(query_string, index_path)
    else:
        return query.query_origin(query_string)


@timing_decorator_with_storage(timing_results)
def rerank(query_string, ref_list, index_path):
    if index_path:
        return query.rerank(query_string, ref_list, index_path)
    else:
        return query.rerank(query_string)


def process_dataset(data_list, tokenizer, use_ref=False, use_rerank=False, use_index=False, index_path=None):
    input_list = []
    for item in tqdm(data_list, desc="RAG Progress"):
        if not use_ref:  # RAG
            if use_index:  # 使用倒排索引
                ref_list = query.query(item['name'], index_path) if index_path else query.query(item['name'])
            else:  # 使用RAG
                query_string = f"生成{item['name']}的小传"
                ref_lists = retrieval(query_string, index_path)
                if use_rerank:
                    ref_lists = rerank(query_string, ref_lists, index_path)
                ref_list = [ref['text'] for ref in ref_lists]
            item['rag'] = ref_list
        else:
            item['rag'] = item['quotes']
            random.shuffle(item['rag'])

        input_text, instruction = get_summary_input(item, item['rag'], tokenizer, return_instruction=True)

        # add to output file
        item['instruction'] = instruction

        input_list.append(input_text)
    return input_list, data_list


def get_dataset(file_path, tokenizer, use_ref, use_rerank, use_index, index_path):
    BASE_DATA_DIR = "data/biography"
    name_list = file_path.split(",") if ',' in file_path else [file_path]
    data_lists = dict()

    for _file in name_list:
        if _file:     
            with open(os.path.join(BASE_DATA_DIR, _file) + ".json", 'r') as f:
                data_list = json.load(f)
            data_input, data_list = process_dataset(data_list, tokenizer, use_ref, use_rerank, use_index)
            data_lists[_file] = (data_list, data_input)
    
    return data_lists


def load_model_and_tokenizer(model_path, lora_path=None):
    # load model
    model_path = os.path.expanduser(model_path)
    lora_path = os.path.expanduser(lora_path) if lora_path else None
    if os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path)
            print(f"[INFO] lora is loaded from {lora_path}!")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # use vllm server
    else:
        model = json.loads(model_path)
        model["model"] = BASE_MODEL if "model" not in model else model["model"]
        tokenizer = AutoTokenizer.from_pretrained(model["model"], trust_remote_code=True)
    return model, tokenizer


def anti_hallucination_infer(model, tokenizer, generation_config, input_data, terminators):
    response = ""
    _buffer = ""
    _references = input_data['rag']
    _generation_config = copy.deepcopy(generation_config)
    check_infos = []

    while True:
        _input = get_summary_input(input_data, _references, tokenizer, response_prefix=response+_buffer)
        res = model_infer(_input, model, tokenizer, _generation_config, terminators=terminators, skip_special_tokens=False)
        _buffer = _buffer + res
        print(f"[INFO] generated:\t{res}\n\n")

        if _buffer[-1] == "，" and len(_buffer) > 10 \
            or _buffer[-1] == "。" \
            or tokenizer.eos_token in _buffer \
            or len(_buffer) >= _generation_config['max_new_tokens']:

            if _buffer == "<|im_end|>" or _buffer == "。<|im_end|>":
                break
            elif _buffer == "。":
                response = response + _buffer
                continue

            _references = filter_documents_by_similarity(_references, _buffer)  # filter refs to easy check
            check_info = hallc_check(_references, input_data['name'], _buffer, model, tokenizer, _generation_config)
            check_infos.append({
                'origin_text': _buffer,
                'hallc': check_info[0],
                'details': check_info[1]
                })

            if check_info[0] == False and _generation_config['temperature'] >= 0.1:  # hallucination detected
                print(f"[DEBUG] hallucination detected: {_buffer}\n{check_info[1]}\n\n")
                # hallucination correction
                _buffer = hallc_correction(_references, _buffer, check_info[1], input_data['name'])
                if _buffer is None:  # hallucination is not solved
                    _generation_config['temperature'] -= 0.1
                    _buffer = ""
                    # if len(response) > 0 and response[-1] in ["，", "。"]:
                    #     response = response[:-1]  # remove last token
                    continue
                else:
                    check_infos[-1]['corrected_text'] = _buffer

            # save and initialize tmp variance
            # response = response + _buffer
            # if tokenizer.eos_token in response:
            #     response = response.split(tokenizer.eos_token)[0]  # remove eos_token
            #     break
            # _references = input_data['rag']
            # _buffer = ""
            # continue

            # # re-generate with filtered refs and lower temperature
            # _references = filter_documents_by_similarity(_references, _buffer)
            # _generation_config['temperature'] = 0.1
            # _buffer = ""  

            
            # save and initialize tmp variance
            response = response + _buffer
            _references = input_data['rag']
            _generation_config['temperature'] = generation_config['temperature']  # reset temperature
            _buffer = ""
            if tokenizer.eos_token in response:
                response = response.split(tokenizer.eos_token)[0]  # remove eos_token
                break

        else:  # continue infer until check point
            print(f"[DEBUG] res[-1]: {res[-1]}")

    return response, check_infos


@timing_decorator_with_storage(timing_results)
def batch_infer(data_list, data_inputs, model, tokenizer, generation_config, hallc_infer):
    responses = []
    for input_data, input_text in tqdm(zip(data_list, data_inputs), desc="Inference Progress"):
        check_infos = None
        # Anti-Hallucination Inference
        if hallc_infer:
            generation_config['max_new_tokens'] = 128
            terminators = [  # 定义终止符
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                tokenizer.encode("。")[0],
                tokenizer.encode("，")[0],
            ]
            response, check_infos = anti_hallucination_infer(model, tokenizer, generation_config, input_data, terminators)        

        else:
            # BUG: the problem of eos/pad token is not fixed yet
            # model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
            # generated_ids = model.generate(
            #     **model_inputs,
            #     generation_config=GenerationConfig(**generation_config),
            #     eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
            # )
            # generated_ids = [
            #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            # ]
            # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = model_infer(input_text, model, tokenizer, generation_config)

        responses.append({
            'response': response,
            'hallc_check': check_infos
        })
        if random.randint(0,9)==0:
            print(f"[INFO] Response:\n{response}")
    return responses


def main(args):
    generation_config = {
        # "n": 1, 
        "temperature": args.temperature,   # 0.3
        # "max_tokens": 3072,  # 2048
        "top_p": 0.9, 
        "top_k": 10, 
        "num_beams": 1,    
        "max_new_tokens": 1024, 
        "do_sample": True,  # try do sample
        # "use_beam_search": False,
        # "best_of": 1  # Add best_of parameter
    }

    # make save directory
    print("save path: ", args.save_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.lora_path)

    # load test datasets
    data_lists = get_dataset(args.file_path, tokenizer, args.use_ref, args.use_rerank, args.use_index, args.index_path)

    for file_name, data_pair in data_lists.items():
        data_list, data_inputs = data_pair
        file_name_for_save = file_name.split(".")[0]
        save_list_path = os.path.join(args.save_dir, f"{file_name_for_save}")
        if not os.path.exists(save_list_path): 
            os.makedirs(save_list_path) 

        responses = batch_infer(data_list, data_inputs, model, tokenizer, generation_config, args.hallc_infer)
        
        results = []
        for dt, res in zip(data_list, responses):
            results.append({
                **dt,
                'response': res['response'],
                'hallc_check': res['hallc_check']
            })

        if not args.use_ref:
            rag_predictions = [item['rag'] for item in data_list]
            rag_references = [item['quotes'] for item in data_list]
            rag_metrics = compute_rag_metric(rag_predictions, rag_references)
        
        # Compute Summarization Metrics
        labels = [item['output'] for item in data_list]
        predictions = [res['response'] for res in responses]
        if 'rouge' in args.metric:
            sum_metrics = compute_rouge(predictions, labels)

        # save model responses
        with open(os.path.join(save_list_path,'output.json'), 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
            print(f"Model Responses are saved in {save_list_path}")

        # save metric results
        with open(os.path.join(save_list_path, 'results.txt'), 'w', encoding='utf-8') as txt_file:
            txt_file.write(f"Test args: {vars(args)}")
            txt_file.write(f"Test time:")
            for func_name, elapsed_time in timing_results.items():
                txt_file.write(f"{func_name}: {sum(elapsed_time)} seconds\n")
            if not args.use_ref:
                txt_file.write(f"RAG Metrics:\n")
                for key, value in rag_metrics.items():
                    txt_file.write(f"{key}: {value}\n")
            txt_file.write(f"Summarization Metrics:\n")
            for key, value in sum_metrics.items():
                txt_file.write(f"{key}: {value*100:.2f}\n")
        print(f"Metric {args.metric} results are saved in {save_list_path}")
      
      
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="An example program with command-line arguments.")
    parser.add_argument('--file_path', default="few-shot/abt_buy,", help='The file path of test datasets.')
    parser.add_argument("--save_dir", default="project/results/", help="The save directory outputs and score.")
    parser.add_argument("--model_path", default=os.path.join(os.path.expanduser("~/pretrain/Qwen2.5-7B-Instruct")), help="The file path of model weight or ip address for vllm")
    parser.add_argument("--lora_path", default=None, help="The file path of lora weights")
    parser.add_argument("--subsize", default="-1", type=int, help="The number of sub-samples.")
    parser.add_argument("--beam_search", action="store_true", help="Whether use the beam search.")
    parser.add_argument("--temperature", default=0.35, type=float, help="The temperature of model.")
    parser.add_argument("--batch_size", default="8", type=int, help="The number of samples in a batch.")
    parser.add_argument("--use_ref", action='store_true', default=False, help="Use ground truth reference rather than rag")
    parser.add_argument("--index_path", default="new_storage_600_ICL", help="The path of index")
    parser.add_argument("--use_rerank", action='store_true', default=False, help="Use rag+reranker")
    parser.add_argument("--use_index", action='store_true', default=False, help="Use index to retrieve documents")
    parser.add_argument("--hallc_infer", action='store_true', default=False, help="Use hallucination_check when inference")
    parser.add_argument("--metric", default="rouge", type=str, help="The metric of evaluation")
    
    args = parser.parse_args()
    main(args)