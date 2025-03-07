
def format_references(references):
    """
    将字符串列表以指定格式连接：
    文献1：\n\n{}\n\n文献2：\n\n{}...
    
    Args:
        references (list): 字符串列表
    
    Returns:
        str: 格式化后的字符串
    """
    formatted_docs = []
    for i, doc in enumerate(references, start=1):
        formatted_docs.append(f"文献{i}：\n\n{doc}")
    return "\n\n".join(formatted_docs)


def get_summary_input(item, ref_list, tokenizer, response_prefix="", return_instruction=False):
    prompt = (
        f"基于历史文献生成“{item['name']}”的小传。综合所有输入文献信息，不得遗漏任何细节。"
        f"忠实于文献内容，不得引入文献外的信息，也不可推测未明确记载的内容。"
        f"按时间顺序详细记录人物的生平经历，包括科举三试（乡试、会试、殿试）成绩、官职履历、著作及重要事件，确保逻辑清晰，信息一致。"
        f"语言风格需仿照文献中的文言文表达，保持凝练、典雅，避免使用现代化词汇或过于文学化的描述。"
        f"确保生成内容涵盖所有文献中的核心信息，避免重复或遗漏。\n"
        # f"### 示例小传：\n吴世涵，字渊若，号霞孙、榕畺、榕江。嘉庆八年（1803）生，浙江处州府遂昌县人。道光八年（1828）乡试举人，道光二十年殿试第三甲第四十二名，赐同进士出身，即用知县。历任云南临安府通海县、大理府太和县知县，充道光二十九年（1849）云南乡试同考官，咸丰二年（1852）调东川府会泽知县。奔父丧，卒于舟中。著有《又其次斋诗集》《平昌诗萃》。\n王国元，字荩臣。贵州贵阳府贵筑县籍，浙江绍兴府诸暨县人。乾隆三十九年（1774）乡试举人，乾隆五十二年殿试第三甲第二十八名，赐同进士出身。分刑部学习期满，历刑部主事、刑部员外郎、吏部郎中。嘉庆十六年（1811）授广西南宁府知府，革职。"
        f"### 历史文献：\n" + format_references(ref_list)
    )
    
    if 'tonggu' in tokenizer.name_or_path.lower():  # TongGu
        system_message = "你是通古，由华南理工大学DLVCLab训练而来的古文大模型。你具备丰富的古文知识，为用户提供有用、准确的回答。"
        text = f"{system_message}\n<用户> {prompt}\n<通古> "
    else:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if return_instruction:  # 直接返回文本
        return text+response_prefix, prompt
    return text + response_prefix