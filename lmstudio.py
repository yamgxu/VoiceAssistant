import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import queue
from aiohttp import ClientSession, ClientTimeout
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    base_url="http://localhost:1234/v1",
    api_key="YOUR_API_KEY_HERE",
)


def stream_openai_chat(text):
    print("Starting chat with OpenAI...")
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        messages=[{"role": "user", "content": text}],
        stream=True,
    )
    print("Chat started. Type 'exit' to quit.")
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""


def get_stream(text, q, loop):
    asyncio.set_event_loop(loop)

    async def run_fetch_data():
        async for data in stream_chat_async(text):
            q.put(data)

    # 运行异步任务
    loop.run_until_complete(run_fetch_data())
    q.put(None)


def stream_chat(text):
    # 创建新事件循环
    q = queue.Queue()
    loop = asyncio.new_event_loop()

    thread = threading.Thread(target=get_stream, args=(text, q, loop,))
    thread.start()
    while True:
        data = q.get()
        if data is None:
            # 如果是结束标记，退出循环
            break
        yield data


async def stream_chat_async(text):
    headers = {
        "content-type": "application/json",
    }
    json_data = {
        "messages": [
            {"role": "system", "content": "你现在开始扮演一个万能助手"},
            {"role": "assistant", "content": "好的"},
            {"role": "user", "content": text}
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": True
    }
    async with ClientSession(headers=headers, timeout=ClientTimeout(total=600, sock_read=600)) as session:
        async with session.post("http://localhost:1234/v1/chat/completions", json=json_data) as response:
            response.raise_for_status()
            async for stream in response.content.iter_any():
                if stream:
                    if b"content" in stream:
                        data = json.loads(stream.decode().split("data: ")[1])
                        yield data["choices"][0]["delta"]["content"]


async def main():
    pass


if __name__ == '__main__':
    message = ""
    for chunk in stream_chat("睡觉睡觉。"):
        message += chunk
        time.sleep(1)
        print(chunk, end="")

    print("\n", message)
