import torch
from torch import nn
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import io
from urllib.request import urlopen
from alibabacloud_imageseg20191230.client import Client
from alibabacloud_imageseg20191230.models import SegmentHairAdvanceRequest
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions
os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'] = 'LTAI5tBero4THZYv8djP55gw'
os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET'] = 'fcZ8RvB9Kk1FSzbJGnnus2nvx8qHg2'

class AvgLabLoss(nn.Module):
    def __init__(self, parsenet):
        super(AvgLabLoss, self).__init__()
        self.parsenet = parsenet
        self.criterion = nn.L1Loss()
        self.M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])

    def gen_hair_mask(self, input_image):
            
        # 阿里分割法
        # 把tensor数据转化为而数据对象
        input_image = input_image.to('cpu')[0]

        input_image = (input_image * 255).round().clamp(0, 255).to(torch.uint8)

        to_pil_image = transforms.ToPILImage()
        pil_image = to_pil_image(input_image)
        
        with io.BytesIO() as buffer:
            original_extension = "JPEG"

            # 保存为PNG，如果原始是JPEG
            save_format = "PNG" if original_extension.lower() == "jpg" else original_extension.upper()
            pil_image.save(buffer, format=save_format)

            binary_data = buffer.getvalue()

        config = Config(
            access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
            access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
            endpoint='imageseg.cn-shanghai.aliyuncs.com',
            region_id='cn-shanghai'
        )

        # img = open(r'test_images/ref_img/5.png', 'rb')

        segment_hair_request = SegmentHairAdvanceRequest()
        with io.BytesIO(binary_data) as image_buffer:
            # 设置 image_urlobject 为虚拟文件对象
            segment_hair_request.image_urlobject = image_buffer

            runtime = RuntimeOptions()

            # 初始化Client
            client = Client(config)
            response = client.segment_hair_advance(segment_hair_request, runtime)
            response_data = response.body
            image_url = response_data.data.elements[0].image_url
            height = response_data.data.elements[0].height
            width = response_data.data.elements[0].width
            x = response_data.data.elements[0].x
            y = response_data.data.elements[0].y
            original_width = 1024
            original_height = 1024

        if image_url:

            image_data = urlopen(image_url).read()

            image = Image.open(io.BytesIO(image_data))
            # 调整位置大小

            # 指定背景色（可以根据需求选择）
            background_color = (0, 0, 0)  # 这里选择黑色

            # 创建一个黑色背景的RGB图像
            rgba_background = Image.new('RGBA', (original_width,original_height), background_color)

            # 把背景和图像转化为numpy
            image_array = np.array(image,dtype=np.uint8)
            rgba_background_array = np.array(rgba_background,dtype=np.uint8)

            #设置阈值a,Alpha 通道值小于某个阈值 a 时，将 RGB 通道全部设置为零
            a = 255  # 你可以根据需要调整阈值
            # 将 Alpha 通道小于阈值的区域的 RGB 通道设置为零
            alpha_channel = image_array[:, :, 3]
            image_array[alpha_channel < a, :3] = 0
            image_array[alpha_channel >= a, :3] = 255

            # 依据y,x,heigth,width得出图像在背景中的位置
            rgba_background_array[y:y + height, x:x + width] = image_array 

            image_pil = Image.fromarray(rgba_background_array, 'RGBA')

            result_image = Image.alpha_composite(Image.new('RGBA', rgba_background.size, (0, 0, 0, 0)), image_pil)

            result_image = np.array(result_image)
            result_image = result_image[:, :, 0]
            tensor_image = torch.from_numpy(result_image)
            tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
            tensor_image = tensor_image.to('cuda:0')
            hair_mask = (tensor_image == 255).float()
            return hair_mask
            
            # 原方法
            # labels_predict = torch.argmax(self.parsenet(input_image)[0], dim=1).unsqueeze(1).long().detach()
            # hair_mask = (labels_predict==10).float()
            # input_image = input_image * hair_mask

            # 加入SLIC
            # denormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
            # input_image = denormalize(input_image[0]).clamp(0, 1)
            # input_image *= hair_mask[0]
            # input_image = input_image.cpu().numpy()
            # input_image = input_image.transpose(1, 2, 0)
            # # 加入SLIC
            # for numSegments in (10,):
            #     # apply SLIC and extract (approximately) the supplied number of segments
            #     segments = slic(input_image, n_segments=numSegments, sigma=2, compactness=0.2)

            #     # 遍历所有超像素块，计算颜色平均值
            #     segment_colors = []
            #     for segment_id in np.unique(segments):
            #         # Create a mask for the current segment
            #         mask = (segments == segment_id)

            #         # 排除全零（黑色）超像素块
            #         if np.any(input_image[mask] != 0):
            #             # 计算颜色平均值
            #             average_color = np.mean(input_image[mask], axis=(0, 1))
            #             segment_colors.append((segment_id, average_color))

            #     # 找到颜色平均值最丰富的超像素块
            #     if segment_colors:
            #         target_segment_id, _ = max(segment_colors, key=lambda x: np.sum(x[1]))

            #         # 保存颜色平均值最丰富的超像素块
            #         target_mask = (segments == target_segment_id)
            #         target_mask_tensor = torch.tensor(target_mask, dtype=torch.bool, device='cuda:0')
            #         target_mask_tensor = target_mask_tensor.view(1, 1, 1024, 1024)
            #         return target_mask_tensor
                

            # labels_predict = torch.argmax(self.parsenet(input_image)[0], dim=1).unsqueeze(1).long().detach()
            # hair_mask = (labels_predict==10).float()
            # return hair_mask
                
    def gen_hair_mask2(self, input_image):
            
        labels_predict = torch.argmax(self.parsenet(input_image)[0], dim=1).unsqueeze(1).long().detach()
        hair_mask = (labels_predict==10).float()
        return hair_mask
            
    def f(self, input):
        output = input * 1
        mask = input > 0.008856
        output[mask] = torch.pow(input[mask], 1 / 3)
        output[~mask] = 7.787 * input[~mask] + 0.137931
        return output

    def rgb2xyz(self, input):
        assert input.size(1) == 3
        M_tmp = self.M.to(input.device).unsqueeze(0)
        M_tmp = M_tmp.repeat(input.size(0), 1, 1)  # BxCxC
        output = torch.einsum('bnc,bchw->bnhw', M_tmp, input)  # BxCxHxW
        M_tmp = M_tmp.sum(dim=2, keepdim=True)  # BxCx1
        M_tmp = M_tmp.unsqueeze(3)  # BxCx1x1
        return output / M_tmp

    def xyz2lab(self, input):
        assert input.size(1) == 3
        output = input * 1
        xyz_f = self.f(input)
        # compute l
        mask = input[:, 1, :, :] > 0.008856
        output[:, 0, :, :][mask] = 116 * xyz_f[:, 1, :, :][mask] - 16
        output[:, 0, :, :][~mask] = 903.3 * input[:, 1, :, :][~mask]
        # compute a
        output[:, 1, :, :] = 500 * (xyz_f[:, 0, :, :] - xyz_f[:, 1, :, :])
        # compute b
        output[:, 2, :, :] = 200 * (xyz_f[:, 1, :, :] - xyz_f[:, 2, :, :])
        return output

    def cal_hair_avg(self, input, mask):
        x = input * mask
        sum = torch.sum(torch.sum(x, dim=2, keepdim=True), dim=3, keepdim=True) # [n,3,1,1]
        mask_sum = torch.sum(torch.sum(mask, dim=2, keepdim=True), dim=3, keepdim=True) # [n,1,1,1]
        mask_sum[mask_sum == 0] = 1
        avg = sum / mask_sum
        return avg
    
    def alinet(self):
        config = Config(
        access_key_id=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID'),
        access_key_secret=os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET'),
        endpoint='imageseg.cn-shanghai.aliyuncs.com',
        region_id='cn-shanghai'
        )

        img = open(r'test_images/ref_img/5.png', 'rb')
        segment_hair_request = SegmentHairAdvanceRequest()
        segment_hair_request.image_urlobject = img
        runtime = RuntimeOptions()

        client = Client(config)
        response = client.segment_hair_advance(segment_hair_request, runtime)
        response_data = response.body
        image_url = response_data.data.elements[0].image_url
        rgb_image = image_url[:, :, :3]
        rgb_image = rgb_image.unsqueeze(0)

        # 将通道维度移到第1个位置，变成(1, 4, 1024, 1024)
        rgba_image = rgba_image.permute(0, 3, 1, 2)
        rgb_image = rgb_image.permute(0, 3, 1, 2)
        return rgb_image
        

    def forward(self, target, result):
        if target.shape[-1] != 1:
            mask_target = self.gen_hair_mask(target)
            target_Lab = self.xyz2lab(self.rgb2xyz((target + 1) / 2.0))
            target_Lab_avg = self.cal_hair_avg(target_Lab, mask_target)
        else:
            target_Lab_avg = self.xyz2lab(self.rgb2xyz((target + 1) / 2.0))

        mask_result = self.gen_hair_mask2(result)
        result_Lab = self.xyz2lab(self.rgb2xyz((result + 1) / 2.0))
        result_Lab_avg = self.cal_hair_avg(result_Lab, mask_result)

        loss = self.criterion(target_Lab_avg, result_Lab_avg)
        return loss