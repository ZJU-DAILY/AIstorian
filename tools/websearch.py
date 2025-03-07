from duckduckgo_search import DDGS

def duckduckgo_search(query: str, max_results: int = 5) -> list:
    with DDGS() as ddgs:
        results = []
        for result in ddgs.text(query, max_results=max_results):
            results.append(result)
        return results

# 示例使用
query = "最新的 Python 技术"
results = duckduckgo_search(query)
for i, result in enumerate(results, start=1):
    print(f"{i}. {result.get('title')} - {result.get('href')}")


def websearch(query: str) -> str:
    results = duckduckgo_search(query=query, max_results=3)
    return results