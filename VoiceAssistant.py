import queue
import concurrent.futures
import lmstudio
from TTS import TTService
from chat import get_text_stream

tts = TTService.TTService('TTS/models/paimon6k.json', 'TTS/models/paimon6k_390000.pth', 'character_paimon', 1.1)
tts.read_play('你好，我是派蒙,。')


def interlocution():
    for text in get_text_stream():
        print(text)
        message = ''
        for chunk in lmstudio.stream_chat(text):
            print(chunk, end='', flush=True)
            indices = [index for index, char in enumerate(chunk) if char in ',，。！？\n']
            if len(indices) > 0:
                index = indices[0]
                message = message + chunk[:index + 1]
                # 播放message
                tts.read_play(message)
                message = chunk[index + 1:]
            else:
                message = message + chunk
        if message != '':
            tts.read_play(message)
        print("\nuser: ", end='')


def get_text_queue(q):
    for text in get_text_stream():
        print("------------"+text)
        q.put(text)
    q.put(None)  # 结束标志


def get_text_stream_from_queue(q):
    while True:
        text = q.get()
        print(text)
        if text is None:
            break
        message = ''
        for chunk in lmstudio.stream_chat(text):
            if not q.empty():
                break
            print(chunk, end='', flush=True)
            indices = [index for index, char in enumerate(chunk) if char in ',，。！？\n']
            if len(indices) > 0:
                index = indices[0]
                message = message + chunk[:index + 1]
                # 播放message
                tts.read_play(message)
                message = chunk[index + 1:]
            else:
                message = message + chunk
        if message != '':
            tts.read_play(message)
        print("\nuser: ", end='')


def chat():
    q = queue.Queue()  # 创建一个队列
    # 创建一个线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(get_text_queue, q)
        executor.submit(get_text_stream_from_queue, q)
    executor.shutdown(wait=True)


if __name__ == '__main__':
    interlocution()
