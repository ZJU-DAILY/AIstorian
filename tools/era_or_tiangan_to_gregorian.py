import re
import json5
from qwen_agent.tools.base import BaseTool, register_tool

JIA_ZI_YEAR = 1984  # 甲子年对应的公历年份之一

@register_tool('era_to_gregorian')
class EraToGregorian(BaseTool):
    description = '将明清年号转换为公历年，支持如 "嘉庆元年"、"嘉庆辛酉"、"同治十三年甲戌"等格式。'

    parameters = [{
        'name': 'era',
        'type': 'string',
        'description': '年号，格式如 "嘉庆元年" 或 "嘉庆辛酉"。',
        'required': True
    }]

    def chinese_to_number(self, chinese_str: str) -> int:
        chinese_numbers = {
            '元': 1, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            '十': 10
        }
        if len(chinese_str) == 1:
            result = chinese_numbers[chinese_str[-1]]
        elif len(chinese_str) == 2 and chinese_str[-2] == '十':
            result = 10 + chinese_numbers[chinese_str[-1]]
        elif len(chinese_str) == 3 and chinese_str[-2] == '十':
            result = chinese_numbers[chinese_str[-1]] + chinese_numbers[chinese_str[-3]] * 10
        else:
            raise ValueError(f"输入格式错误{chinese_str}，请使用格式：汉字数字，例如 '一' 或 '十'")
        return result
    
    def tiangan_dizhi_to_year(self, base_year, tiangan, dizhi):
        # 定义天干地支的固定顺序列表
        gan_zhi_list = [
            "甲子", "乙丑", "丙寅", "丁卯", "戊辰", "己巳", "庚午", "辛未", "壬申", "癸酉",
            "甲戌", "乙亥", "丙子", "丁丑", "戊寅", "己卯", "庚辰", "辛巳", "壬午", "癸未",
            "甲申", "乙酉", "丙戌", "丁亥", "戊子", "己丑", "庚寅", "辛卯", "壬辰", "癸巳",
            "甲午", "乙未", "丙申", "丁酉", "戊戌", "己亥", "庚子", "辛丑", "壬寅", "癸卯",
            "甲辰", "乙巳", "丙午", "丁未", "戊申", "己酉", "庚戌", "辛亥", "壬子", "癸丑",
            "甲寅", "乙卯", "丙辰", "丁巳", "戊午", "己未", "庚申", "辛酉", "壬戌", "癸亥"
        ]
        
        # 检查输入的天干和地支是否有效
        if tiangan+dizhi not in gan_zhi_list:
            raise ValueError(f"Invalid tiangan&dizhi: {tiangan+dizhi}")
        
        # 计算天干和地支的索引
        tiangan_index = gan_zhi_list.index(tiangan+dizhi)

        # 计算对应的公历年
        year = base_year + tiangan_index
        
        return year

    def call(self, params: str, **kwargs) -> str:
        era_or_tiangan = json5.loads(params)['era']

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

        year_numbers = {
            '元': 1, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            '十': 10
        }

        # 天干地支对应的基础公历年份
        tian_gan = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
        di_zhi = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
        base_tg = '甲'
        base_dz = '子'
        cycle = 60  # 一个天干地支周期为60年

        original_era_or_tiangan = era_or_tiangan
        if era_or_tiangan.startswith('明') or era_or_tiangan.startswith('清'):
            era_or_tiangan = era_or_tiangan[1:]
        if '科' in era_or_tiangan:
            era_or_tiangan = era_or_tiangan.split('科')[0]
        if '年' in era_or_tiangan:  # 年份后跟天干地支时，仅通过年份判断
            era_or_tiangan = era_or_tiangan.split('年')[0]

        # 先前两个字符作为年号
        era_name = era_or_tiangan[:2]

        if era_name in era_start_years_qing:
            start_year = era_start_years_qing.get(era_name)
        elif era_name in era_start_years_ming:
            start_year = era_start_years_ming.get(era_name)
        elif era_or_tiangan[0] in tian_gan:  # 只有天干地支纪年，缺少年号
            start_year = 0
            era_or_tiangan = "年号" + era_or_tiangan
        else:
            raise ValueError(f"未知的年号：{era_name}，可能是只有天干地支纪年，缺少年号")

        # 年号后年份
        year_name = era_or_tiangan[2:]

        # 判断是否为天干地支年份格式
        is_tiangan_dizhi = False
        if len(year_name) >= 2:
            tg, dz = year_name[0], year_name[1]
            if tg in tian_gan and dz in di_zhi:
                is_tiangan_dizhi = True

        is_era_year = False
        # 判断是否每个字符都是数字
        if all(char in year_numbers for char in year_name):
            is_era_year = True

        if not is_era_year and not is_tiangan_dizhi:
            raise ValueError(f"{era_or_tiangan}中{year_name}年份格式错误，请使用汉字数字或干支纪年，例如 '元年' 或 '辛酉'")

        if not is_tiangan_dizhi:
            year_number = self.chinese_to_number(year_name)  # 将中文数字转换为阿拉伯数字
            year_offset = year_number - 1  # 减1是因为起始年为1年
            gregorian_year = start_year + year_offset 

        else:
            tg, dz = year_name[0], year_name[1]

            if tg not in tian_gan or dz not in di_zhi:
                raise ValueError(f"{year_name}输入格式错误，请使用格式：天干+地支，例如 '甲子'")
            
            if start_year:            
                base_year = JIA_ZI_YEAR - (JIA_ZI_YEAR - start_year + cycle - 1) // cycle * cycle
                gregorian_year = self.tiangan_dizhi_to_year(base_year, tg, dz)
                if gregorian_year < start_year:
                    gregorian_year += cycle
            else:
                year_offset = self.tiangan_dizhi_to_year(0, tg, dz)
                gregorian_year = [jia_zi_year + year_offset for jia_zi_year in [1564, 1624, 1684, 1744, 1804, 1864]]
                gregorian_year = f"由于缺少年号，可能的年份包括：{', '.join(map(str, gregorian_year))}"

        return json5.dumps({'era': original_era_or_tiangan, 'gregorian_year': gregorian_year}, ensure_ascii=False)


# 测试样例
if __name__ == '__main__':
    era_to_gregorian = EraToGregorian()
    print(era_to_gregorian.call('{"era": "癸亥年"}'))
    print(era_to_gregorian.call('{"era": "癸亥"}'))