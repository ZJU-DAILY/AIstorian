import csv

def load_address_dictionary(file_path: str) -> dict:
    address_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                short_name = row[0].strip()
                full_name = row[1].strip()
                address_dict[short_name] = full_name
    return address_dict

def get_full_address(short_name: str, address_dict: dict) -> str:
    return address_dict.get(short_name, "未找到对应的全称")

# 示例使用
file_path = "address_dict.csv" 
address_dict = load_address_dictionary(file_path)
def address_query(query):
    if query.endswith("县"):
        query = query[:-1]
    full_name = get_full_address(query, address_dict)
    return full_name