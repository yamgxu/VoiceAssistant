#
import torch
from funasr import AutoModel

model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common",disable_update=True)

res = model.generate(
    input="output3.wav"
)
res2 = model.generate(
    input="output1.wav"
)
tensor1=res[0]["spk_embedding"]
tensor2=res2[0]["spk_embedding"]
euclidean_dist = torch.dist(tensor1,tensor2 , p=2)

print("欧几里得距离:", euclidean_dist.item())
