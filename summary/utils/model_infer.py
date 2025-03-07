from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import re
import os


BASE_URL = "http://IP:8848/v1"
MODEL = os.path.join(os.path.expanduser("~/pretrain/Qwen2.5-7B-Instruct"))
API_KEY = "sk"

LLM_CFG = {
    # 使用dashscope
    # 'model': 'qwen-max-2025-01-25',
    # 'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    # 'api_key': 'sk',

    # 使用vllm
    'model': os.path.join(os.path.expanduser("~/pretrain/Qwen2.5-7B-Instruct")),
    'model_server': 'http://IP:8880/v1',  # base_url, also known as api_base
    'api_key': 'EMPTY',
    'generate_cfg': {
        'temperature': 0.1,  #0.5
        'top_p': 0.8,
        'max_tokens': 4000,
    }   
}

def model_infer(input_text, model, tokenizer, generation_config, **kwargs):
    # 定义终止符
    # terminators = kwargs.get("terminators", [tokenizer.convert_tokens_to_ids("<|im_end|>")])
    
    if isinstance(model, dict) or model is None:
        if model is None:
            model = {
                "base_url": LLM_CFG["model_server"],
                "model": LLM_CFG["model"]
            }
        client = OpenAI(
            base_url=model["base_url"],
            api_key=model.get("api_key", LLM_CFG["api_key"]),
        )

        if isinstance(input_text, list):
            messages = input_text
        elif '<|im_start|>' in input_text:
            input_text = input_text + '<|im_end|>'
            pattern = r'<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>'
            matches = re.findall(pattern, input_text, re.DOTALL)
            messages = [
                {"role": match[0], "content": match[1].strip()}
                for match in matches if len(match[1].strip()) > 0
            ]
        else:
            messages = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                         {"role": "user", "content": input_text}]

        completion = client.chat.completions.create(
            model=model["model"],
            messages=messages,
            extra_body={
                "include_stop_str_in_output": True,
                # "stop_token_ids": terminators,
                "top_p": generation_config.get("top_p", 0.8),
                "temperature": generation_config.get("temperature", 0.5)
            }
        )

        print(completion.choices[0].message)
        return completion.choices[0].message.content
    
    else:
        terminators = kwargs.get("terminators", [tokenizer.eos_token_id])
        model_inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
        generated_ids = model.generate(
            **model_inputs,
            **generation_config,
            return_dict_in_generate=True,
            eos_token_id=terminators
        )
        # print(f"[BEGIN]{tokenizer.decode(generated_ids['sequences'][0], skip_special_tokens=False)}\n\n")
        return tokenizer.decode(generated_ids['sequences'][0][len(model_inputs.input_ids[0]):], skip_special_tokens=kwargs.get("skip_special_tokens", True))
    