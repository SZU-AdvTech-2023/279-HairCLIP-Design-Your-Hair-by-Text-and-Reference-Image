import torch
from models.stylegan2.model import Generator
from models.face_parsing.model import BiSeNet

def load_base_models():
    ckpt = "pretrained_models/ffhq.pt"
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    mean_latent = torch.load(ckpt)["latent_avg"].unsqueeze(0).unsqueeze(0).repeat(1,18,1).clone().detach().cuda()

    seg_pretrained_path = 'pretrained_models/seg.pth'
    seg = BiSeNet(n_classes=16) # BiSeNet源代码中设置了19，表示模型在图像分割任务中可以区分的不同类别的数量。
    # 在这里，指定输出类别数为 16 表明模型被训练以对图像进行 16 类别的分割，不知道具体是哪16类
    seg.load_state_dict(torch.load(seg_pretrained_path), strict=False) # strict=False 允许在加载权重时跳过不匹配的键
    for param in seg.parameters():
        param.requires_grad = False # 将所有模型参数的梯度计算设置为 False，因为在推理阶段不需要进行梯度更新。
    seg.eval() # seg.eval()：将模型设置为推理模式，这会关闭一些在训练时使用的特定于训练的层，例如批量归一化的统计信息不再更新。
    seg = seg.cuda()

    return g_ema, mean_latent, seg