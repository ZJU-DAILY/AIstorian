import os
from llama_index.core import StorageContext, load_index_from_storage
from FlagEmbedding import FlagReranker
import pickle
from .load_index import load_or_create_index
# PERSIST_DIR = "./retrieval/new_storage"
PERSIST_DIR = ""
my_file_path = ""


def query(key):
    # 读取倒排索引
    with open('new_index\inverted_index.pkl', 'rb') as f:
        inverted_index = pickle.load(f)
    
    # 读取关键词关系
    with open('new_index\key_relations.pkl', 'rb') as f:
        key_relations = pickle.load(f)
    
    # 读取文档存储
    with open('new_index\segments.pkl', 'rb') as f:
        segments = pickle.load(f)
    
    related_keys = key_relations.get(key, set([key]))
    related_keys.add(key)
    result_ids = set()
    for kw in related_keys:
        result_ids.update(inverted_index.get(kw, []))
    
    # 获取相关文档
    result_docs = [segments[doc_id] for doc_id in result_ids]
    return result_docs

if __name__ == "__main__":
    query("黄机")