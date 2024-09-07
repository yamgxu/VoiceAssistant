import ollama
ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': "你好，我是派蒙,。"}],
    stream=False,
)
def stream_chat(text):
    stream = ollama.chat(
            model='llama3.1',
            messages=[{'role': 'user', 'content': text}],
            stream=True,
        )

    for chunk in stream:
        yield chunk['message']['content']
