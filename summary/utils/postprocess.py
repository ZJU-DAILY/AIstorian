import json
import re

def extract_json_content(model_response):
    try:
        # 假设 JSON 内容以 `{` 开头并以 `}` 结尾
        start_index = model_response.find("{")
        end_index = model_response.rfind("}")
        if start_index != -1 and end_index != -1:
            json_str = model_response[start_index:end_index + 1]
            json_content = json.loads(json_str)  # 将 JSON 字符串解析为 Python 字典
            return json_content
        else:
            raise ValueError("未找到 JSON 格式的内容")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解码失败: \n{model_response}\n{e}")

def terminator_strip(text):
    pattern = r'<start>(.*?)<\\end>'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    elif "如下：" in text:
        return text.split("如下：")[1].strip()
    elif "如下（" in text:
        return text.split("）：")[1].strip()
    elif "文本：" in text:
        return text.split("文本：")[1].strip()
    else:
        return text