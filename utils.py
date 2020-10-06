# 导入包
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from PIL import Image
import numpy as np
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
from torch.utils.data.dataset import Dataset

use_cuda = torch.cuda.is_available()

# 定义好读取图像的Dataset

class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 设置最长的字符长度为6个
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (6 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:6]))
    def __len__(self):
        return len(self.img_path)

# 测试集数据扩增（Test Time Augmentation，简称TTA）也是常用的集成学习技巧，数据扩增不仅可以在训练时候用，而且可以同样在预测时候进行数据扩增。

def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None

    # TTA 次数
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3, c4, c5 = model(input)
                c0 = c0.cpu()
                c1 = c1.cpu()
                c2 = c2.cpu()
                c3 = c3.cpu()
                c4 = c4.cpu()
                c5 = c5.cpu()
                output = np.concatenate([
                    c0.data.numpy(),
                    c1.data.numpy(),
                    c2.data.numpy(),
                    c3.data.numpy(),
                    c4.data.numpy(),
                    c5.data.numpy()
                ],axis=1)
                test_pred.append(output)
        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    return test_pred_tta