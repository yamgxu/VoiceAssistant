import asyncio
import time
import numpy as np

import websockets
from aiohttp import web

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

import ollamachat
from TTS import TTService

tts = TTService.TTService('TTS/models/paimon6k.json', 'TTS/models/paimon6k_390000.pth', 'character_paimon', 1.1)
tts.read_play('你好，我是派蒙,。')

chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", disable_update=True, device="cpu")
model_dir = "iic/SenseVoiceSmall"

model1 = AutoModel(
    model=model_dir,
    # vad_model="fsmn-vad",
    # vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
    disable_update=True

)

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
BITS_PER_SAMPLE = 16

clients = set()
is_receiving_data = False  # 标志位，控制是否接收数据
input_str = asyncio.Queue()  # 控制命令
user_input = asyncio.Queue()  # 控制命令队列


# 对音频数据进行增益处理
def apply_gain(data, gain):
    # 将二进制数据转换为 int16
    audio_data = np.frombuffer(data, dtype=np.int16)

    # 增加音频增益
    audio_data = audio_data * gain

    # 防止溢出，限制到 int16 的范围
    audio_data = np.clip(audio_data, -32768, 32767)

    # 将处理后的音频数据转换回二进制
    return audio_data.astype(np.int16).tobytes()


def text_to_wav(text):
    print(text)
    message = ''
    for chunk in ollamachat.stream_chat(text):
        print(chunk, end='', flush=True)
        indices = [index for index, char in enumerate(chunk) if char in ',，。！？\n']
        if len(indices) > 0:
            index = indices[0]
            message = message + chunk[:index + 1]
            # 播放message
            file_path = "outwav/output" + str(time.time()) + '.wav'
            tts.read_save(message, file_path, tts.hps.data.sampling_rate)
            yield file_path
            message = chunk[index + 1:]
        else:
            message = message + chunk
    if message != '':
        file_path = "outwav/output" + str(time.time()) + '.wav'
        tts.read_save(message, file_path, tts.hps.data.sampling_rate)
        yield file_path


async def get_reply():
    global input_str, is_receiving_data
    while True:
        for file_path in text_to_wav(await user_input.get()):
            # wav = AudioSegment.from_wav(file_path)
            # 导出为MP3文件\
            # s = str(random.randint(1000, 9999))
            # mp3_file = "outmp3/" + time.strftime("%Y%m%d%H%M%S", time.localtime()) +s+ ".mp3"
            # wav.export(mp3_file, format="mp3")
            # os.remove(file_path)
            await input_str.put("http://192.168.68.239:8080/" + file_path)
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.1)


async def control_device():
    global is_receiving_data, input_str
    while True:
        status = await input_str.get()
        try:
            if status == "exit":
                break
            if status == "start":
                # 控制逻辑：发送 'start' 以开始接收数据
                websockets.broadcast(clients, "start")
                print("Sent 'start' to ESP32-S3")
            if status == "stop":
                websockets.broadcast(clients, "stop")
                print("Sent 'stop' to ESP32-S3")
            if status.startswith("http"):
                websockets.broadcast(clients, status)
                print("Sent mp3 file to ESP32-S3")
                await asyncio.sleep(1)
                continue
        except websockets.ConnectionClosed:
            print("control_device closed")
        await asyncio.sleep(0.1)


def buffer_to_text(data):
    res = model1.generate(
        input=data,
        cache={},
        language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  #
        merge_length_s=15,
        disable_pbar=True,
    )
    return rich_transcription_postprocess(res[0]["text"])


async def receive_audio_data(websocket: websockets.WebSocketServerProtocol):
    global is_receiving_data, user_input
    clients.add(websocket)
    input_str.put_nowait("http://192.168.68.239:8080/outwav/output1726548550.1189485.wav")

    while True:
        # 接收来自 ESP32-S3 的数据，并保存到文件中
        # wav_file = wave.open("received_audio" + str(time.time()) + ".wav", 'wb')
        # wav_file.setnchannels(NUM_CHANNELS)
        # wav_file.setsampwidth(BITS_PER_SAMPLE // 8)
        # wav_file.setframerate(SAMPLE_RATE)
        try:
            print("Started receiving data.")
            alldata = []
            cache = {}
            beg = -1
            CHUNK = 16
            limitation = 0
            async for message in websocket:
                # 写入数据到 wav 文件中
                message = apply_gain(message, 10)

                if beg > -1:
                    pass
                    # wav_file.writeframes(message)
                data = message
                if len(data) > 0:
                    if len(alldata) > 0:
                        alldata = alldata + data
                    else:
                        alldata = data
                    is_final = False
                    res = model.generate(disable_pbar=True, input=data, cache=cache, is_final=is_final,
                                         chunk_size=200)
                    if len(res[0]["value"]):
                        for i in res[0]["value"]:
                            beg = i[0] if i[0] > -1 else beg
                            end = i[1]
                            if end == -1:
                                pass
                            else:
                                print(f"start{beg},{end}")
                                tdata = alldata[beg * CHUNK * 2 - limitation:end * CHUNK * 2 - limitation]
                                if len(tdata) < 10:
                                    continue
                                await websocket.send("stop")
                                await user_input.put(buffer_to_text(tdata))
                                break
                if len(alldata) > CHUNK * 2 * 1000 * 100:
                    alldata = alldata[1000 * 50 * CHUNK * 2 - limitation:]
                    limitation = 1000 * 50 * CHUNK * 2

        except websockets.ConnectionClosed:
            print("Connection closed")
            clients.remove(websocket)
            break  # 连接关闭时，结束循环
        finally:
            pass
            # wav_file.close()
        await asyncio.sleep(0.1)  # 控制接收频率，避免占用过多资源


async def handler(websocket: websockets.WebSocketServerProtocol):
    print("New connection")
    # 创建两个并发任务：一个用于控制 ESP32-S3，一个用于接收数据
    await asyncio.gather(
        receive_audio_data(websocket),
    )
    # HTTP 处理函数
async def handle(request: web.Request) -> web.Response:
    file_path =  request.rel_url.path
    print(file_path)
    try:
        with open(file_path[1:], 'rb') as file:
            return web.Response(body=file.read(), content_type='audio/wav')
    except FileNotFoundError:
        return web.Response(status=404, text='File not found')

async def init_http_server() -> web.Application:
    app = web.Application()
    app.router.add_get('/{filename:.+}', handle)
    return app


async def main():
    # 启动 WebSocket 服务器
    websocket_server = websockets.serve(handler, "0.0.0.0", 8088, ping_interval=1000, ping_timeout=2000, max_size=1000 * 16)

    # 启动 HTTP 服务器
    http_app = await init_http_server()
    runner = web.AppRunner(http_app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()

    print("HTTP server started on port 8080")

    # 启动 WebSocket 服务器
    print("WebSocket server started on port 8088")
    async with websocket_server:
        await asyncio.Future()  # 保持服务器运行

async def run_all_tasks():
    results = await asyncio.gather(
        main(),
        control_device(),
        get_reply()
    )
    print("All tasks completed")
    print("Results:", results)


if __name__ == "__main__":
    asyncio.run(run_all_tasks())
