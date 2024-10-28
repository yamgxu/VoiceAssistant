import ollama

ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': "你好，我是派蒙,。"}],
    stream=False,
)


def stream_chat(text):
    stream = ollama.chat(
        model='llama3.1',
        messages=[{"role": "system", "content": "你现在开始扮演一个万能助手,请用中文跟我聊天吧。"},
                  {'role': 'user', 'content': text}],
        stream=True,
    )

    for chunk in stream:
        yield chunk['message']['content']


async def async_stream_chat(text):
    stream = await ollama.AsyncClient().chat(model='llama3.1',
                                             messages=[{'role': 'user', 'content': text}],
                                             stream=True
                                             )
    async for chunk in stream:
        yield chunk['message']['content']
