import re
import json5
from qwen_agent.tools.base import BaseTool, register_tool


@register_tool('qing_era_to_gregorian')
class QingEraToGregorian(BaseTool):
    description = '将明、清代年号转换为公历年，输入格式为：年号+中文数字年份，例如 "康熙五十年"。'
    
    parameters = [{
        'name': 'era',
        'type': 'string',
        'description': '清代或明代年号及年份，格式如 “康熙五十年”',
        'required': True
    }, {
        'name': 'dynasty',
        'type': 'string',
        'description': '朝代名称, 可选值为 "qing" 或 "ming"',
        'required': False
    }]

    def chinese_to_number(self, chinese_str: str) -> int:
        chinese_numbers = {
            '元': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            '十': 10, '百': 100, '千': 1000, '万': 10000
        }

        result = 0
        unit = 1
        temp = 0
        
        for char in reversed(chinese_str):
            if char in chinese_numbers:
                num = chinese_numbers[char]
                if num == 10 or num == 100 or num == 1000 or num == 10000:
                    if temp == 0:
                        temp = 1
                    result += temp * num
                    temp = 0
                else:
                    temp += num * unit
                    unit *= 10
            else:
                raise ValueError(f"非法字符：{char}")

        return result + temp

    def call(self, params: str, **kwargs) -> str:
        # 解析输入参数
        era = json5.loads(params)['era']
        dynasty = json5.loads(params).get('dynasty', 'qing')  # 默认为清代

        # 清代年号对应的公历起始年
        era_start_years_qing = {
            "顺治": 1644,
            "康熙": 1662,
            "雍正": 1723,
            "乾隆": 1736,
            "嘉庆": 1796,
            "道光": 1821,
            "咸丰": 1851,
            "同治": 1862,
            "光绪": 1875,
            "宣统": 1909
        }

        # 明代年号对应的公历起始年
        era_start_years_ming = {
            "洪武": 1368,
            "建文": 1399,
            "永乐": 1403,
            "宣德": 1425,
            "正统": 1435,
            "景泰": 1450,
            "成化": 1465,
            "弘治": 1487,
            "正德": 1506,
            "嘉靖": 1522,
            "隆庆": 1567,
            "万历": 1572,
            "泰昌": 1620,
            "天启": 1621,
            "崇祯": 1628
        }

        if era.startswith('明'):
            dynasty = 'ming'
            era = era[1:]
        elif era.startswith('清'):
            dynasty = 'qing'
            era = era[1:]

        if dynasty == 'qing' or dynasty == '清':
            era_start_years = era_start_years_qing
        elif dynasty == 'ming' or dynasty == '明':
            era_start_years = era_start_years_ming
        else:
            raise ValueError(f"未知的朝代：{dynasty}")
        
        match = re.split(r'年', era)
        if match:
            era = match[0] + '年'  # 返回‘年’及之前的部分
        elif not era.endswith('年'):
            era = era + '年'
        elif era.endswith('科'):
            era = era[:-1] + '年'

        # 正则匹配年号格式
        era = re.sub(r'[^\u4e00-\u9fa5]', '', era)
        match = re.match(r"^(洪武|建文|永乐|宣德|正统|景泰|成化|弘治|正德|嘉靖|隆庆|万历|泰昌|天启|崇祯|顺治|康熙|雍正|乾隆|嘉庆|道光|咸丰|同治|光绪|宣统)([\u4e00-\u9fa5]+)年$", era)
        if not match:
            raise ValueError(f"{era}输入格式错误，请使用格式：年号+年份，例如 '康熙五十年'")
        era_name, year_chinese = match.groups()
        year_number = self.chinese_to_number(year_chinese)  # 将中文数字转换为阿拉伯数字
        year_offset = year_number - 1  # 减1是因为起始年为1年
        start_year = era_start_years.get(era_name)
        if start_year is None:
            raise ValueError(f"未知的年号：{era_name}")

        # 计算并返回公历年份
        gregorian_year = start_year + year_offset
        return json5.dumps({'dynasty': dynasty, 'era': era, 'gregorian_year': gregorian_year}, ensure_ascii=False)
