import json
from rapidfuzz import fuzz
import jieba
from summary.utils.model_infer import model_infer
from summary.utils.formatter import format_references
from summary.utils.postprocess import extract_json_content

# 分词处理
def tokenize(text):
    return " ".join(jieba.lcut(text))
    
# 计算分词后的相似度
def compute_similarity(documents, claim):
    tokenized_claim = tokenize(claim)
    similarity_scores = []
    for doc in documents:
        tokenized_doc = tokenize(doc)
        similarity_score = fuzz.token_set_ratio(tokenized_doc, tokenized_claim)
        similarity_scores.append((doc, similarity_score))
    return similarity_scores

def filter_documents_by_similarity(documents, claim, top_n=3, threshold=50):
    """
    筛选与句子部分相似度得分最高的文档。

    Args:
        documents (list): RAG检索到的文档列表。
        claim (str): 检测为幻觉的句子。
        top_n (int): 返回得分最高的前N个文档。
        threshold (float): 最小相似度得分阈值。

    Returns:
        list: 筛选后的文档列表。
    """
    # 计算每个文档与句子的相似度得分
    # similarity_scores = [
    #     (doc, fuzz.partial_ratio(doc, claim)) for doc in documents
    # ]
    similarity_scores = compute_similarity(documents, claim)
    
    # 筛选得分大于或等于阈值的文档
    filtered_docs = [(doc, score) for doc, score in similarity_scores if score >= threshold]
    
    # 按得分降序排序
    filtered_docs.sort(key=lambda x: x[1], reverse=True)
    
    if not filtered_docs:  # TODO: 当所有文档都不满足阈值条件时，说明该事实为幻觉，包括无中生有和文献冲突
        # 如果没有文档满足阈值条件，则返回所有文档
        # return documents
        # 当所有文档的得分都小于阈值时，返回相似度最高的一个文档
        highest_doc = max(similarity_scores, key=lambda x: x[1])
        return [highest_doc[0]]
    
    # 返回得分最高的前N个文档
    return [doc for doc, _ in filtered_docs[:top_n]]


def hallc_check(docs, character, claim, model, tokenizer, generation_config):
    """Check whether the claim is induced from references"""

    prompt = '''请评估以下文献内容是否支持所陈述的关于进士“{character}”的小传文本，并按照指定格式返回结果。

### 文献信息：
{references}

### 小传文本：
{claim}

### 错误类型：
1. 文献未提及：参考文献中未提及小传文本内容，例如人物的科举、官职等未在参考文献中提及。例如：
    - 小传文本中提及的事实在参考文献中未提及。
    - 小传文本中事实发生的年份在参考文献中未提及。

2. 文献否定：参考文献中的信息明确否定了小传文本中的内容。例如：
    - 参考文献明确指出小传文本中人物科举的名次错误。
    - 参考文献明确指出小传文本中人物科举的年代错误。
    - 参考文献明确指出小传文本中人物的籍贯错误。
    - 参考文献明确指出小传文本中人物的官职错误。
    - 参考文献明确指出小传文本中人物的科举的名次错误。
    - 参考文献明确指出小传文本中人物的科举的年代错误。

### 分析过程：
1. 分析小传文本包含的信息；
2. 分析参考文献与小传文本相关的信息；
3. 判断小传文本中的信息是否被参考文献支持，如果参考文献给出了更详细的内容，不影响小传文本的正确性。
4. 如果小传文本的内容被参考文献支持，请以JSON格式返回: {{"answer": "支持"}}。即使参考文献中还有更多关于该人物的详细信息，也请返回支持。
5. 如果发现任何不支持之处，请详细列出这些差异，并以JSON格式返回: {{"answer": "不支持", "details": "错误类型及具体不一致的地方"}}。'''
    prompt  = prompt.format(character=character, references=format_references(docs), claim=claim)

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    generation_config = {
        'max_new_tokens': 512,
        'temperature': 0.8,
        'top_p': 0.95,
        'top_k': 50,
        'num_return_sequences': 1,
    }

    max_attempts = 3  # 最大尝试次数
    attempt = 0
    while attempt < max_attempts:
        try:
            res = model_infer(text, None, tokenizer, generation_config)  # BUG: use base model by api.
            print(f"[INFO] hallucination check result: {res}")
            response = extract_json_content(res)
            break  # 如果成功，不再尝试
        except Exception as e:
            attempt += 1
            generation_config['temperature'] = generation_config['temperature'] - 0.1
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt == max_attempts:
                print("All attempts failed. Exiting.")
                return (False, None)
        
    return (response.get("answer") == "支持", response.get('details'))

