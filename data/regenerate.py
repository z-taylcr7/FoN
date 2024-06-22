from openai import OpenAI
import unicodedata
import numpy as np
import json

# 使用你的 API 密钥进行身份验证
client = OpenAI(
    api_key="sk-JP4PAEcXezqNgIsyF6F9846109Fa412aB0F9B76fFc9f487f",
    base_url="https://lonlie.plus7.plus/v1",
)
# 创建一个聊天模型
chat_model = "gpt-3.5-turbo"


# 开始一个新的聊天会话
def chat_with_bot(message):
    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
    )

    # 返回机器人的回答
    content = response.choices[0].message.content
    return unicodedata.normalize("NFKC", content)


raw_data = []
with open('ra_gsm8k.jsonl', 'r') as f:
    for line in f:
        raw_data.append(json.loads(line))


data = []
for i, item in enumerate(raw_data):
    line = {}
    line['prompt'] = item['prompt']
    rsteps = len(item['response'].split('\n'))
    question = (
        item['prompt']
        + "Please answer in no more than"
        + str(np.floor(rsteps / 2))
        + " reasoning steps with least redundance. Step 1."
    )
    ans = chat_with_bot(question)
    line['response'] = ans
    if i % 100 == 0:
        print(i, item['prompt'], ans)
    data.append(line)

with open('regen_gsm8k.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')
