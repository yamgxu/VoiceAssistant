import wave

import pyaudio
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1              # 单声道
RATE = 16000              # 采样率
CHUNK = 16          # 每次读取的音频块大小
audio = pyaudio.PyAudio()


stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
frames = []  # 用于存储音频数据

for i in range(0, int(RATE / CHUNK * 5)):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音结束")

# 停止并关闭音频流
stream.stop_stream()
stream.close()
audio.terminate()
OUTPUT_FILE = "output4.wav"  # 输出文件名

# 将录制的音频数据保存到文件
wf = wave.open(OUTPUT_FILE, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))  # 将 frames 列表中的数据写入文件
wf.close()

print(f"音频已保存到 {OUTPUT_FILE}")
