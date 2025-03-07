import json5
from qwen_agent.tools.base import BaseTool, register_tool

@register_tool('tiangan_dizhi_to_gregorian')
class TianganDizhiToGregorian(BaseTool):
    description = '将天干地支年份转换为公历年，输入格式为：天干+地支，例如 "辛酉"。'

    parameters = [{
        'name': 'tiangan_dizhi',
        'type': 'string',
        'description': '天干地支年份，格式如 "甲子"。',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        # 解析输入参数
        tiangan_dizhi = json5.loads(params)['tiangan_dizhi']

        # 天干和地支的列表
        tian_gan = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
        di_zhi = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
        
        # 解析输入的天干地支年份
        if len(tiangan_dizhi) != 2:
            raise ValueError(f"{tiangan_dizhi}输入格式错误，请使用格式：天干+地支，例如 '甲子'")
        
        tg, dz = tiangan_dizhi[0], tiangan_dizhi[1]
        
        if tg not in tian_gan or dz not in di_zhi:
            raise ValueError(f"{tiangan_dizhi}输入格式错误，请使用格式：天干+地支，例如 '甲子'")
        
        # 计算天干地支对应的数字
        tg_index = tian_gan.index(tg)
        dz_index = di_zhi.index(dz)
        
        # 计算天干地支年份的差值
        base_year = 1984  # 甲子年对应的公历年份
        base_tg = '甲'
        base_dz = '子'
        
        # 计算天干地支的周期
        cycle = 60  # 一个天干地支周期为60年
        
        # 计算天干地支的年份偏移
        offset = (tg_index - tian_gan.index(base_tg)) + (dz_index - di_zhi.index(base_dz))
        
        if offset < 0:
            offset += cycle
        
        # 计算公历年份
        gregorian_year = base_year + offset
        
        return json5.dumps({'tiangan_dizhi': tiangan_dizhi, 'gregorian_year': gregorian_year}, ensure_ascii=False)
