import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  #attempted relative import with no known parent package
from retrieval.utils import query_test
from summary.metrics.rouge import compute_rag_metric



BASE_DATA_DIR = "data/biography"
with open(os.path.join(BASE_DATA_DIR, "test") + ".json", 'r') as f:
    data_list = json.load(f)

pred = []
refer = []

for item in data_list:
    refer.append(item['quotes'])
    my_query = "生成"+item['name']+"的小传"
    res = query_test.query(my_query)
    final_res = query_test.rerank(my_query,res)
    curr_pred = []
    for i in final_res:
        curr_pred.append(i['text'])
    pred.append(curr_pred)

metrics = compute_rag_metric(pred,refer)
for key,value in metrics.items():
    print(f'{key}:{value}')
