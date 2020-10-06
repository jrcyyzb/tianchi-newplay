# 导入包
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
from torch.utils.data.dataset import Dataset

from Module import *
from utils import *

test_path = glob.glob('./input/mchar_test_a/*.png')
# print(test_path[0])
test_path.sort()
# print(test_path[0])
# print(len(test_path))
test_label = [[1]] * len(test_path)

# print(len(test_label))

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                           transforms.Compose([
                               transforms.Resize((64, 128)),
                               transforms.RandomCrop((60, 120)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ])),
    batch_size=40,
    shuffle=False,
)

test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
# print(len(test_label))
# print(test_label)

# 加载预训练的模型
model = SVHN_Model1()
if use_cuda:
    model.cuda()
model.load_state_dict(torch.load('./model.pt'))

test_predict_label = predict(test_loader, model, 1)
test_predict_label = np.vstack([
    test_predict_label[:, :11].argmax(1),
    test_predict_label[:, 11:22].argmax(1),
    test_predict_label[:, 22:33].argmax(1),
    test_predict_label[:, 33:44].argmax(1),
    test_predict_label[:, 44:55].argmax(1),
    test_predict_label[:, 55:66].argmax(1),
]).T

test_label_pred = []
for x in test_predict_label:
    test_label_pred.append(''.join(map(str, x[x!=10])))

import pandas as pd
df_submit = pd.read_csv('./input/mchar_sample_submit_A.csv')
df_submit['file_code'] = test_label_pred
df_submit.to_csv('renset18.csv', index=None)