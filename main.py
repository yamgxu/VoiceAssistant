import torch
from funasr import AutoModel

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming",device="cpu",disable_update=True)
model1 = AutoModel(model="ct-punc",device="cpu",disable_update=True)
res = model1.generate(input="今天的会就到这里吧")


import pyaudio
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1              # 单声道
RATE = 16000              # 采样率
CHUNK = 9600              # 每次读取的音频块大小
audio = pyaudio.PyAudio()
RECORD_SECONDS = 5000        # 录音时间


stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")
cache={}
text = ""
texto = 0
end=''
re_is_final = True
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    is_final = False
    res = model.generate(disable_pbar=True,input=data, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
    text=text+res[0]["text"]
    if res[0]["text"]=='':
        if  re_is_final:
            is_final = True
            res = model.generate(disable_pbar=True,input=data, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
            text=text+res[0]["text"]
            re_is_final = False
            cache={}
    else:
        re_is_final = True

    if text != "" and res[0]["text"]!="":
        #print(text)
        res = model1.generate(disable_pbar=True,input=text)
        #print(res[0]["text"])
        (text_,end) = (res[0]["text"][:-1],res[0]["text"][-1]) if res[0]["punc_array"][-1:][0]>1 else (res[0]["text"],"")
        print(text_[texto:].replace("？","？\n").replace("。","。\n").replace("！","！\n"),end="")
        texto = len(text_)
#
        tensor = res[0]["punc_array"][:-1]
        # 查找大于2的元素
        greater_than_two = tensor > 2
        # 获取大于2的元素的下标
        if greater_than_two.sum()<1:
            continue
        indices = torch.where(greater_than_two)[-1][0]
        text=text[indices+1:]
        texto = len(text)
    else:
        if end!="":
            print(end)
            end=""
            text=""
            texto = len(text)
        print(".")






