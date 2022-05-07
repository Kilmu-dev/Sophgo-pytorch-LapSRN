import torch
from lapsrn import Net

model = Net()
model = torch.load('lapsrn_model_epoch_80.pth', map_location=torch.device('cpu')) #保存的训练模型
model.eval()#切换到eval（）
example = torch.rand(64, 1, 3, 3)#生成一个随机输入维度的输入
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("LspSRN_epoch_80.pt")