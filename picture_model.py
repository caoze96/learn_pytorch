import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,datasets
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
import datetime

# 数据准备
transforms_train = transforms.Compose([transforms.ToTensor()])
transforms_vaild =transforms.Compose([transforms.ToTensor()])

ds_train = datasets.ImageFolder('./data/cifar2/train/',
                               transform = transforms_train,
                               target_transform = lambda t:torch.tensor([t]).float())
ds_valid = datasets.ImageFolder('./data/cifar2/test/',
                               transform = transforms_vaild,
                               target_transform=lambda t:torch.tensor([t]).float())

# print(ds_train.class_to_idx)

dl_train = DataLoader(ds_train,batch_size=50,shuffle=True,num_workers=3)
dl_valid = DataLoader(ds_valid,batch_size=50,shuffle=True,num_workers=3)


#
# plt.figure(figsize=(8,8))
# for i in range(9):
#     img,label = ds_train[i]
#     img = img.permute(1,2,0)
#     ax=plt.subplot(3,3,i+1)
#     ax.imshow(img.numpy())
#     ax.set_title("label = %d"%label.item())
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y

net = Net()
# print(net)

# 输出模型的参数
# import torchkeras
# torchkeras.summary(net,input_shape=(3,32,32))

# 训练模型
model = net
model.optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
model.loss_func = torch.nn.BCELoss()
model.metric_func = lambda y_pred,y_true: roc_auc_score(y_true.data.numpy(), y_pred.data.numpy())
model.metric_name = "auc"

def train_step(model,fetures,labels):
    # 训练模式，dropout层发生作用
    model.train()

    # 梯度清零
    model.optimizer.zero_grad()

    # 正向传播求损失
    predictions = model(fetures)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(),metric.item()

def valid_step(model,features,labels):
    model.eval()

    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions,labels)
        metric = model.metric_func(predictions,labels)

    return loss.item(),metric.item()

# features,labels = next(iter(dl_train))
# train_step(model,features,labels)

def train_model(model,epoches,dl_train,dl_valid,log_step_frep):
    metric_name = model.metric_name
    dfhistory = pd.DataFrame(columns=["epoch","loss",metric_name,"val_Loss","val_"+metric_name])
    print("Start Training...")
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("============"*8 + "%s"%nowtime)

    for epoch in range(1,epoches+1):
        # 1,循环训练
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step,(features,labels) in enumerate(dl_train, 1):
            loss,metric = train_step(model, features, labels)

            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step%log_step_frep == 0:
                print(("[step = %d] loss: %.3f,"+metric_name+": %.3f") %
                      (step, loss_sum/step, metric_sum/step))

        # 2.循环验证
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features,labels) in enumerate(dl_valid, 1):
            val_loss, val_metric = valid_step(model,features,labels)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3.记录日志
        info = (epoch,loss_sum/step, metric_sum/step,
                val_loss_sum/val_loss, val_metric_sum/val_step)
        dfhistory.loc[epoch-1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f," + metric_name + "= %.3f,val_loss = %.3f," + "val_" + metric_name + "=%.3f")%info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "=========="*8 + "%s"%nowtime)

    print('Finished Training...')
    return dfhistory

epochs = 20
dfhistory = train_model(model,epochs,dl_train,dl_valid,log_step_frep=50)

