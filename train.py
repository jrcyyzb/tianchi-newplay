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


# 定义好训练数据和验证数据的Dataset
train_path = glob.glob('./input/mchar_train/*.png')
# print(train_path)
train_path.sort()
# print(train_path)
train_json = json.load(open('./input/mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json]
print(len(train_path), len(train_label))

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=40,
    shuffle=True,
)

# for data in train_loader:
#     # print(data[0].size())
#     print(data[1].size())
#     print(data[1])
#     break

val_path = glob.glob('./input/mchar_val/*.png')
val_path.sort()
val_json = json.load(open('./input/mchar_val.json'))
val_label = [val_json[x]['label'] for x in val_json]
print(len(val_path), len(val_label))

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([
                    transforms.Resize((60, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])),
    batch_size=40,
    shuffle=False,
)

# 迭代训练和验证模型
model = SVHN_Model1()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)
best_loss = 1000.0

if use_cuda:
    model = model.cuda()

# 设置训练epochs
epochs = 10



for epoch in range(epochs):
    train_loss = train(train_loader, model, criterion, optimizer)
    val_loss = validate(val_loader,  model, criterion)

    val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
    val_predict_label = predict(val_loader, model, 1)
    print(val_predict_label.size())
    val_predict_label = np.vstack([
        val_predict_label[:, :11].argmax(1),
        val_predict_label[:, 11:22].argmax(1),
        val_predict_label[:, 22:33].argmax(1),
        val_predict_label[:, 33:44].argmax(1),
        val_predict_label[:, 44:55].argmax(1),
        val_predict_label[:, 55:66].argmax(1)
    ]).T
    val_label_pred = []
    for x in val_predict_label:
        val_label_pred.append(''.join(map(str, x[x!=10])))

    val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

    print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
    print(val_char_acc)
    # 记录下验证集精度
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './model.pt')
