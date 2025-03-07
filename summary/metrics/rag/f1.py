# mrr.py
import datasets
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def jaccard_similarity_ngrams(str1, str2, n=2):
    # 提取字符 n-grams
    ngrams1 = {str1[i:i+n] for i in range(len(str1) - n + 1)}
    ngrams2 = {str2[i:i+n] for i in range(len(str2) - n + 1)}
    
    # 计算交集和并集
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    
    # 检查并集是否为空以避免除以零的错误
    if len(union) == 0:
        return 0.0  # 或者返回 None，根据需求选择
    # 计算 Jaccard 相似度
    return len(intersection) / len(union)

@datasets.utils.file_utils.add_start_docstrings("This metric computes Precision, Recall and F1.")
class GroupMetric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="Precision, Recall and F1",
            citation="",
            inputs_description="Predictions and references",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string")),
                    "references": datasets.Sequence(datasets.Value("string")),
                }
            ),
            reference_urls=[],
        )


    def _compute(self, predictions, references, labels=None, pos_label=1, average="binary", sample_weight=None, zero_division="warn"):
        p_list = []
        r_list = []
        f_list = []
        ap_list = []
        for pred, ref in zip(predictions, references):
            pred_binary = []
            curr_p_list = []
            # refer_binary = [1 for i in ref]
            for i in pred:
                curr = 0
                for j in ref:
                    curr = jaccard_similarity_ngrams(i, j)
                    if curr > 0.5:
                        pred_binary.append(1)
                        curr_p_list.append(pred_binary.count(1)/len(pred_binary))
                        break
                if curr <= 0.5:
                    pred_binary.append(0)

            
            if len(pred_binary) == 0 or len(ref) == 0:
                raise ValueError('the length of references or predctions cannot be zero')
            
            p_score = pred_binary.count(1)/len(pred_binary)
            p_list.append(p_score)
            r_score = pred_binary.count(1)/len(ref)
            r_list.append(r_score)
            f_score = 2 * p_score * r_score / (p_score+r_score) if p_score+r_score != 0 else 0  # BUG: 分母为0报错  
            f_list.append(f_score)
            # print(pred_binary)
            # print(curr_p_list)
            ap_list.append(0 if len(curr_p_list) == 0 else sum(curr_p_list)/len(curr_p_list))

        return {"avg_precision": None if len(p_list) == 0 else sum(p_list)/len(p_list),
                "avg_recall": None if len(r_list) == 0 else sum(r_list)/len(r_list),
                "avg_f1": None if len(f_list) == 0 else sum(f_list)/len(f_list),
                "total_precision": p_list,
                "total_recall": r_list,
                "total_f1": f_list,
                "MAP": None if len(ap_list) == 0 else sum(ap_list)/len(ap_list)
                }
    
        
