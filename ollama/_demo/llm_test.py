import requests

# 全局配置
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen:7b"

def get_answer(question:str):
    prompt=f"""
    你是大模型应用开发助手，仅简洁回答技术问题，不知道就回复「暂无相关资料」，禁止编造内容。
    用户问题：{question}
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream":False
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    res_data = resp.json()
    print("接口返回详情：", res_data)
    return res_data["response"]

if __name__=="__main__":
    res = get_answer("什么是Prompt工程？")
    print("模型输出：", res)
