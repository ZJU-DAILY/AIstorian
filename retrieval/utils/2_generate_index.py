import pickle
from collections import defaultdict
from summary.utils.model_infer import model_infer
from tqdm import tqdm
import re


key_relations = {}
segments = []

def generate_prompt_key(text):
    prompt = f"""
        以下是一段用古文书写的文本，其中介绍了关键人物，并且该人物不是书籍的作者或者编者。

        注意: 仅根据输入文本内容判断，不要增加任何外部信息或推测。
        如果存在人物字、号等别名关系，请以“姓名”-“姓氏”-“别名”格式给出回答，如果存在多个别名关系，请以“；”分隔。文中可能出现通假字，例如“子”可能通“字”，请仔细甄别。
        在提取人物名时辨别并忽略如“夫子印”、“老夫子”等古代官职、尊号，确保最后提取出的人物名符合原文语义。
        如果不存在别名关系，则只返回该人物姓名。
        文段可能出现多个人物名字，请根据文段介绍内容决定关键人物，可以列出不同人物，比如有的人物可能只提及姓名但并未过多介绍，则不是关键人物。
        如果存在父子、祖孙等家庭关系，需要根据姓氏补全姓名作为关键人物。

        ### 示例：
        输入文本：
        （清）朱汝珍《词林辑略》卷二：“康熙十二年癸丑科：徐潮，字雪崖，号浩轩，又号青来，浙江钱塘人。散馆，授检讨。官至吏部尚书。追谥文敬。

        输出：
        徐潮-徐-雪崖；徐潮-徐-浩轩；徐潮-徐-青来


        ### 输入文本：

        {text}
        """
    return prompt

def add_to_relation(a,b):
    if a in key_relations:
        key_relations[a].add(b)
    else:
        key_relations[a] = set([b])

def process_response_key(response):
    final_set = set()
    relations = response.strip().split('；')
    for relation in relations:
        names = relation.split('-')
        final_set.add(names[0])
        if len(names) > 2 and names[-1] != '':
            new_name = names[1]+names[-1]
            final_set.add(new_name)
            add_to_relation(names[0],new_name)
            add_to_relation(new_name,names[0])
            
    return final_set

def process_regex(segment, idx):
    curr_block = {}
    names = []
    pattern = [r'“[\u4e00-\u9fa5]+[，]([^，]+)，字([^，]+)，号([^，]+)，又号([^，]+)',
            r'“[\u4e00-\u9fa5]+[，]([^，]+)，字([^，]+)，号([^，。]+)',
            r"([\u4e00-\u9fa5]+)，字([\u4e00-\u9fa5]+)"]
    for p in pattern:
        match = re.search(p, segment)
        if match:
            res = match.groups()
            # last_name = res[0][:2] if len(res[0]) > 3 else res[0][:1]
            for i in res:
                names.append(i)
                if len(names) > 1 and names[-1] != '':
                    add_to_relation(names[0],i)
                    add_to_relation(i,names[0])
                
            break
    curr_block['id'] = idx
    curr_block['keys'] = names
    return curr_block

def generate_blocks(segments):
    # model, tokenizer = load_model_and_tokenizer(model_path)
    
    blocks = []
    idx = 0
    for segment in tqdm(segments, desc="Generate blocks using llm"):
        
        curr_block = process_regex(segment,idx)
        if len(curr_block['keys']) == 0:
            prompt = generate_prompt_key(segment)
            generation_config = {
                'temperature': 0.35,
                'top_p': 0.8,
                'max_tokens': 4000,
            }
            response = model_infer(prompt,None,None,generation_config)
            key_set = process_response_key(response)
            curr_block['id'] = idx
            curr_block['keys'] = key_set
        blocks.append(curr_block)
        idx += 1
    return blocks

def generate_inverted_index(text_blocks):
    # 构建倒排索引
    inverted_index = defaultdict(set)
    for block in text_blocks:
        for keyword in block["keys"]:
            inverted_index[keyword].add(block["id"])
    with open('inverted_index.pkl', 'wb') as f:
        pickle.dump(inverted_index, f)
    return inverted_index

if __name__ == "__main__":
    with open("new_index\segments.pkl", 'rb') as f:
        segments = pickle.load(f)
    blocks = generate_blocks(segments)
    with open("new_index\key_relations.pkl", 'wb') as f:
        pickle.dump(key_relations, f)
    index = generate_inverted_index(blocks)