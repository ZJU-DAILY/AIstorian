import os
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel
from llama_parse import LlamaParse
from llama_index.core.schema import TextNode, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser
import re
from PyPDF2 import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from tqdm import tqdm
from summary.utils.model_infer import model_infer


embed_model = HuggingFaceEmbedding(model_name="")

# [Warning] Test mode in lyl local
# embed_model = HuggingFaceEmbedding(model_name="TencentBAC/Conan-embedding-v1")
# local_path = "../pretrain/conan"
# embed_model = AutoModel.from_pretrained(local_path)
Settings.embed_model = embed_model
Settings.llm = None
# import chromadb
# from llama_index.vector_stores.chroma import ChromaVectorStore

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-"
model_path = ""

generation_config = {
    "n": 1, 
    "temperature": 0.35,  # 0.3
    "max_tokens": 3072, # 2048
    "top_p": 0.9, 
    "top_k": 10, 
    "num_beams": 1,
    "max_new_tokens": 1024, 
    "do_sample": True, # try do sample
    "use_beam_search": False,
    "best_of": 1 # Add best_of parameter
    # "repetition_penalty": 1.3   # harmful to performance !!!
}

# def generate_document(file_path):
#     parser = LlamaParse(
#         result_type="markdown",  # "markdown" and "text" are available
#         # language=Language.SIMPLIFIED_CHINESE,
#         verbose=True,
#         num_workers=1  # 根据文件数量进行设置
#     )
#     return parser.load_data(file_path)
def generate_document(file_path):
    reader = PdfReader(file_path)
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text()
    all_text = all_text.replace("\n", " ")
    all_text = re.sub(r'\s+', " ", all_text)
    all_text = all_text.strip()
    return [Document(text=all_text)]

# def document_to_node(documents):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=100,
#         chunk_overlap=20,
#         length_function=len,
#         is_separator_regex=False,
#         separators=[
#             "\n\n",
#             "\n",
#             # " ",
#             ".",
#             ",",
#             "\uff0c",  # Fullwidth comma
#             "\u3001",  # Ideographic comma
#             "\uff0e",  # Fullwidth full stop
#             "\u3002",  # Ideographic full stop
#             "",
#         ]
#     )
#     parser = LangchainNodeParser(text_splitter)
#     return parser.get_nodes_from_documents(documents)

def document_to_node(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\u201D",
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            
        ]
    )
    parser = LangchainNodeParser(text_splitter)
    return parser.get_nodes_from_documents(documents)

def split_node_llm(nodes):
    # model, tokenizer = load_model_and_tokenizer(model_path)
    res = []
    for node in tqdm(nodes, desc="Splitting nodes using llm"):
        text = node.text
        prompt = generate_prompt(text)
        # response = model_infer(prompt,model,tokenizer,generation_config)
        generation_config = {
            'temperature': 0.35,
            'top_p': 0.8,
            'max_tokens': 4000,
        }
        response = model_infer(prompt,None,None,generation_config)
        final_segments = process_response(response)
        for i in final_segments:
            res.append(i)
    return res



def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def generate_prompt(text):
    prompt = f"""
        以下是一段用古文书写的文本，其中依次介绍了多个人物的生平或事迹。请按顺序对内容进行划分，使每个文本块完整地介绍一个人物。确保每个文本块包含该人物的所有重要信息，不要删除、增加和修改任何文字，不要进行任何古文翻译解释。

        注意: 仅根据输入文本内容判断，不要增加任何外部信息或推测，同时保证所有文本都在划分范围内；如果相同的名字在不同地方提及，认为这些句子各自属于不同文本块，需要分别列出而不要合在一起；主要介绍人物名字可能出现在句子任何地方，请仔细甄别。

        ### 输入文本：

        {text}

        ### 输出格式：
        1. {{文本块1}}
        2. {{文本块2}}
        ...
        """
    return prompt

def process_response(response):
    segments = re.split(r'\d+\.\s', response)  # 按数字加点和空格分割
    segments = [seg.strip() for seg in segments if seg.strip()]  # 去除空白和空段
    return segments


if __name__ == "__main__":
    file_path = ""
    my_documents = generate_document(file_path)
    nodes = document_to_node(my_documents)
    my_nodes = split_node_llm(nodes)