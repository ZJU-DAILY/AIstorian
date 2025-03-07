import json5
import json
import sys
import re
import os
import pprint
from qwen_agent.agents import Assistant, Router
from qwen_agent.tools.base import BaseTool, register_tool
sys.path.append('/home/lfy/Biography')
from summary.utils.postprocess import extract_json_content, terminator_strip

# 定义明、清代年号转公历工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from agents.qing_era_to_gregorian import QingEraToGregorian
# from agents.tiangan_dizhi_to_gregorian import TianganDizhiToGregorian
from tools.era_or_tiangan_to_gregorian import EraToGregorian

from utils.model_infer import LLM_CFG

def init_era_to_gregorian_agent(llm_cfg):
    tools = ['era_to_gregorian']
    system_instruction = '''你是一个乐于助人的AI帮手。
    在收到用户的请求后，你应该：
    - 分析文献名称，判断是否包含年号。
    - 调用工具计算年号对应的公历年。
    - 如果包含年号，当输入参数仅有天干地支时，请将该年号与天干地支一起作为输入参数。
    - 如果文本中的公历年份有误，或者缺少公历年份，请以括号的形式将公历年份插入原文本的年号之后，给出纠正后的文本，使用<start>和<\end>包裹。'''
    bot_era_or_tiangan_to_gregorian = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                name="年号标注帮手",
                description="该帮手的作用是标注文本中年号对应的公历年份。")
    return bot_era_or_tiangan_to_gregorian


def init_era_conflict_agent(llm_cfg):
    tools = ['era_to_gregorian']
    system_instruction = '''你是一个乐于助人的AI帮手。
    在收到用户的请求后，你应该：
    - 根据小传文本中的年号，调用工具计算年号对应的公历年。请注意调用工具的输入必须是皇帝年号+天干地支纪年或者皇帝年号+年份的格式，皇帝年号出现在文献名称中，请一并分析。
    - 判断年份冲突的事件是否一致，不同事件的年份不同不视为冲突。例如，不同科举（'乡试'、'会试'、'殿试'），不同官职（'举人'、'进士'、'翰林院庶吉士'）的年份不同不视为冲突。
    - 如果小传文本中的公历年份有误，或者缺少公历年份，请以括号的形式将公历年份插入原文本的年号之后，给出纠正后的文本，使用<start>和<\end>包裹。'''
    bot_era_or_tiangan_to_gregorian = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                name="年份冲突帮手",
                description="该帮手的作用是解决小传文本与参考文献中年份之间的冲突。")
    return bot_era_or_tiangan_to_gregorian


def init_missing_agent(llm_cfg):
    system_instruction = '''你是一个乐于助人的AI帮手。
在收到用户的请求后，你应该根据用户给定的分析过程，逐步分析，当需要分析年号时，调用‘era_to_gregorian’工具。判断参考文献是否提及小传文本中的信息，并以JSON格式返回结果。'''
    bot_missing = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                name="文献未提及判断帮手",
                description="该帮手的作用是判断参考文献是否提及小传文本中的信息。")
    return bot_missing


def init_name_title_conflict_agent(llm_cfg):
    system_instruction = '''你是一个乐于助人的AI帮手。
    在收到用户的请求后，你应该：
    - 识别文献中的人物名和字号。
    - 如果同一人物在不同文献中出现不同的字号或字名，请合并这些信息，保持一致性。
    - 返回纠正后的文本，使用<start>和<\end>包裹。'''

    bot_name_title_conflict = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        name="字号冲突纠正帮手",
        description="该帮手的作用是检查并纠正文言文小传文本中人物的字号冲突。"
    )
    
    return bot_name_title_conflict


def init_exam_conflict_agent(llm_cfg):
    system_instruction = '''你是一个乐于助人的AI帮手。
在收到用户的请求后，你应该：
- 分析小传文本和参考文献中包含的人物科举经历（乡试、会试、殿试）。
- 逐个判断参考文献中人物的科举信息与小传文本是否冲突，例如乡试的科举年份或者科举名次冲突、会试的科举年份或者科举名次冲突、殿试的科举年份或者科举名次冲突。
- 注意三试（乡试、会试、殿试）是三次考试，他们的年份和名次不同不视为冲突。
- 注意参考文献的权威性，如果对于同一个科举考试，参考文献给出了不同年份或者不同名次，请全部保留。
- 如果确实存在冲突，请根据参考文献纠正小传文本，最终的小传文本使用<start>和<\end>包裹。'''
    
    bot_exam_conflict = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        name="科举经历冲突纠正帮手",
        description="该帮手的作用是检查并纠正文言文小传中人物科举经历（乡试、会试、殿试）的冲突。"
    )
    
    return bot_exam_conflict


def init_position_conflict_agent(llm_cfg):
    system_instruction = '''你是一个乐于助人的AI帮手。
在收到用户的请求后，你应该：
- 分别分析参考文献和小传文本中人物的职务和年份。
- 如果冲突的职务存在时间上的差异，例如在不同年份考取“武举人”与“武进士”、“工科进士”与“翰林院庶吉士”，该差异符合正常的职务晋升过程，不视为职务冲突。
- 如果参考文献中人物的职务与小传文本内容不一致，请以参考文献为准。
- 提供纠正后的文本，使用<start>和<\end>包裹。'''
    
    bot_position_inconsistency = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        name="职务冲突纠正帮手",
        description="该帮手的作用是检查并纠正文言文小传中人物职务的冲突。"
    )
    
    return bot_position_inconsistency


def init_reference_conflict_agent(llm_cfg):
    system_instruction = '''你是一个乐于助人的AI帮手。
在收到用户的请求后，你应该：
- 判断不一致信息中提到的文献与小传文本的冲突是否属实，如果存在冲突，请以参考文献为准纠正小传文本，使用<start>和<\end>包裹。
- 判断不一致信息中提到的文献之间的冲突是否属实，如果存在文献间冲突，请全部保留，纠正小传文本使用<start>和<\end>包裹。'''
    
    bot_reference_conflict = Assistant(
        llm=llm_cfg,
        system_message=system_instruction,
        name="文献冲突纠正帮手",
        description="该帮手的作用是检查并纠正参考文献之间的冲突。"
    )
    
    return bot_reference_conflict


def init_router(llm_cfg):
    # tools = ['era_to_gregorian']
    system_instruction = '''你有下列帮手：
'年份冲突帮手'：该帮手的作用是解决存在年份冲突或不一致的问题。
'科举冲突帮手'：该帮手的作用是解决科举冲突问题。
'字号冲突帮手'：该帮手的作用是解决字号冲突问题。
'文献冲突帮手'：该帮手的作用是解决文献之间存在冲突的问题。(注意是文献之间，不是文献与小传文本之间)
'其他冲突帮手'：该帮手的作用是解决其他冲突问题。

请根据错误信息选择其中一个来帮你回答，选择的模版如下：
Call: ... # 选中的帮手的名字，必须在['文献冲突帮手', '年份冲突帮手', '科举冲突帮手', '字号冲突帮手', '其他冲突帮手']中选，不要返回其余任何内容。
'''
    bot_router = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                name="帮手调用帮手",
                description="该帮手的作用是根据错误类别调用对应的帮手。")
    return bot_router


bot_router = init_router(LLM_CFG)

# Define era agent
bot_era_to_gregorian = init_era_to_gregorian_agent(LLM_CFG)
bot_missing = init_missing_agent(LLM_CFG)
bot_name_title_conflict = init_name_title_conflict_agent(LLM_CFG)
bot_position_conflict = init_position_conflict_agent(LLM_CFG)
bot_reference_conflict = init_reference_conflict_agent(LLM_CFG)
bot_era_conflict = init_era_conflict_agent(LLM_CFG) 
bot_exam_conflict = init_exam_conflict_agent(LLM_CFG)
bot = Router(
    llm=LLM_CFG,
    agents=[bot_missing, bot_name_title_conflict, bot_position_conflict, bot_reference_conflict, bot_era_conflict, bot_exam_conflict],
)


def bot_run(bot, prompt):
    messages = [{'role': 'user', 'content': prompt}]
    is_function_call = False
    is_agent_call = False
    for response in bot.run(messages=messages):
        pass
    for res in response:
        print(f"[INFO] hallc_correction agent: {res}")
        if res.get('name'):
            is_agent_call = True
    if len(response) > 1:
        is_function_call = True
    return response[-1]['content'], is_agent_call


def hallc_correction(references, input_text, hallc_info, character):
    '''根据参考文献和错误信息对人物小传进行纠正'''
    if '未提及' in str(hallc_info):
        return None

    if isinstance(hallc_info, list):
        hallc_details = hallc_info[0]
    if isinstance(hallc_info, dict):
        try:
            hallc_details = hallc_info.get('具体不一致的地方')
        except:
            try:
                hallc_details = hallc_info.get('具体内容')
            except:
                hallc_details = str(hallc_info)
    else:
        hallc_details = str(hallc_info)
    assistant= bot_run(bot_router, str(hallc_details))[0].split('# ')[0].strip()
    print(f"[INFO] assistant call: {assistant}")


    if '字号冲突帮手' in assistant:
         # 去除错误信息中的年份，即括号中的内容，因为他们可能是幻觉
        hallc_details = re.sub(r'（[^）]*）', '', str(hallc_details))
        # print(hallc_details)
        prompt = f'''请你将不一致信息中提到的与小传文本不一致的字号合并到小传文本中，因为一个人可能同时有多个字号。使用<start>和<\end>包裹最终合并后的文本。

### 参考文献：
{references}

### 不一致信息：
{hallc_details}

### 小传文本：
<start>{input_text}<\end>

### 解决过程：
'''
        res, _ = bot_run(bot_name_title_conflict, prompt)
        # print(res)

    elif '年份冲突帮手' in assistant:
        prompt = f'''请你根据不一致信息判断年份是否存在冲突。如果存在冲突，请以括号的形式将公历年份插入原文本的年号之后，给出纠正后的文本，使用<start>和<\end>包裹。

### 参考文献：
{references}

### 不一致信息：
{hallc_details}

### 小传文本：
<start>{input_text}<\end>

### 解决过程：
'''
        global bot_era_conflict
        res, _ = bot_run(bot_era_conflict, prompt)
        # print(res)
    
    elif '科举冲突帮手' in assistant:
        prompt = f'''请你根据不一致信息判断科举经历是否存在冲突。如果存在冲突，请以参考文献为准纠正小传文本，使用<start>和<\end>包裹。

### 参考文献：
{references}

### 不一致信息：
{hallc_details}

### 冲突原则：
- 三试（乡试、会试、殿试）是三次考试，不同科举考试的年份和名次不同不视为冲突。
- 如果参考文献中人物的职务与小传文本内容不一致，请以参考文献为准。
- 在得出冲突的结论前，先判断参考文献中的考试是否与小传文本中的考试是同一个考试（乡试、会试、殿试）。
- 注意参考文献的权威性，如果对于同一个科举考试，参考文献给出了不同年份或者不同名次，请全部保留。

### 小传文本：
<start>{input_text}<\end>

### 解决过程：
'''
        global bot_exam_conflict
        res, _ = bot_run(bot_exam_conflict, prompt)
        # print(res)

    elif '文献冲突帮手' in assistant:
        prompt = f'''请你判断不一致信息中提到的文献是否与小传文本存在冲突。如果存在冲突，请以参考文献为准纠正小传文本，使用<start>和<\end>包裹。

### 参考文献：
{references}

### 不一致信息：
{hallc_details}

### 小传文本：
<start>{input_text}<\end>

### 解决过程：
'''
        global bot_reference_conflict
        res, _ = bot_run(bot_reference_conflict, prompt)
        # print(res)

    else:
        print(f"[INFO] assistant: {assistant}")
        return None