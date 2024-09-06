import io
import sys

import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch" ,disable_update=True ,device="cpu")
model_dir = "iic/SenseVoiceSmall"

model1 = AutoModel(
    model=model_dir,
    #vad_model="fsmn-vad",
    #vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
    disable_update=True

)


import pyaudio
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1              # 单声道
RATE = 16000              # 采样率
CHUNK = 16          # 每次读取的音频块大小
audio = pyaudio.PyAudio()


stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

def get_text():
    print("Recording...")
    cache={}
    beg=0
    limitdata=0
    alldata= []
    while True:
        data = stream.read(CHUNK*200)
        if len(alldata)>0:
            alldata=alldata+data
        else:
            alldata=data
        is_final = False
        res =model.generate(disable_pbar=True,input=data, cache=cache, is_final=is_final, chunk_size=200)
        if len(res[0]["value"]):
            for i in res[0]["value"]:
                beg =i[0] if i[0] > -1 else beg
                end =i[1]
                if end == -1:
                    pass
                else:
                    print(f"start{beg},{end}")
                    tdata=alldata[beg*CHUNK*2-limitdata:(end)*CHUNK*2-limitdata]
                    if len(tdata) < 10:
                        continue
                    res = model1.generate(
                        input=tdata,
                        cache={},
                        language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                        use_itn=True,
                        batch_size_s=60,
                        merge_vad=True,  #
                        merge_length_s=15,
                        disable_pbar=True,
                    )
                    yield rich_transcription_postprocess(res[0]["text"])






def get_text_stream():
    print("Recording...")
    cache={}
    alldata= []

    beg=0
    limitdata=0
    while True:
        data = stream.read(CHUNK*200)
        if len(alldata)>0:
            alldata=alldata+data
        else:
            alldata=data

        is_final = False
        res =model.generate(disable_pbar=True,input=data, cache=cache, is_final=is_final, chunk_size=200)
        if len(res[0]["value"]):
            for i in res[0]["value"]:
                beg =i[0] if i[0] > -1 else beg
                end =i[1]
                if end == -1:
                    pass
                else:

                    print(f"start{beg},{end}")
                    print(f"alldata{len(alldata)}")

                    tdata=alldata[beg*CHUNK*2-limitdata:(end)*CHUNK*2-limitdata]

                    if len(tdata) < 10:
                        continue
                    res = model1.generate(
                        input=tdata,
                        cache={},
                        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                        use_itn=True,
                        batch_size_s=60,
                        merge_vad=True,  #
                        merge_length_s=15,
                        disable_pbar=True,
                    )
                    yield rich_transcription_postprocess(res[0]["text"])
                    alldata=alldata[end*CHUNK*2-limitdata:]
                    limitdata=end*CHUNK*2




if __name__ == '__main__':
    for text in get_text_stream():
        print("user:", text)

