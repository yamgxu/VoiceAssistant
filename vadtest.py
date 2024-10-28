import time

from funasr import AutoModel

chunk_size = 200 # ms


#model = AutoModel(model="fsmn-vad", model_revision="v2.0.4",disable_update=True)
model = AutoModel(model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch" ,disable_update=True ,device="cpu")

wav_file = "received_audio1726490989.0058498.wav"
import soundfile

speecha, sample_rate = soundfile.read(wav_file)
print(f"Sample rate: {sample_rate}")
CHUNK = 16

cache = {}
data=[]
for i in range(0,speecha/1024 -1):
    speech = speecha[i*1024:(i+1)*1024]
    time.sleep(0.1)
    while True:
        if len(speech) >= CHUNK * 200:
            data = speech[:CHUNK * 200]
        else:
            break
        res = model.generate(disable_pbar=True,input=data, cache=cache, is_final=False, chunk_size=chunk_size)
        speech = speech[CHUNK * 200:]
        if len(res[0]["value"]):
            print(res)