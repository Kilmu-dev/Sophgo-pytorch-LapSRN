import torch
from lapsrn import Net

model = Net()
model = model.load_state_dict(torch.load('lapsrn_model_epoch_100.pth', map_location=torch.device('cpu')))#保存的训练模型
model.eval()#切换到eval（）
example = torch.rand(1, 3, 320, 480)#生成一个随机输入维度的输入
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("LspSRN.pt")