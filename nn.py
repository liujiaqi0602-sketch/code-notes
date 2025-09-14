from random import sample

from torch.utils.data import Dataset
from tqdm import tqdm
from data_loader import one_dataloader

import os
import sys
import pandas as pd
import numpy as np
import  torch.nn as nn
import torch.optim as optim

#初始化神经网络模型
class NN(nn.Module):
    def __init__(self,input_dim,hidden_dim1,hidden_dim2,output_dim):
        super().__init__()
        self.layer1=nn.linear(input_dim,hidden_dim1)
        self.layer2=nn.linear(hidden_dim1,hidden_dim2)
        self.layer3=nn.linear(hidden_dim2,output_dim)
    def forward(self,x):
        x =self.layer1(x)
        x =self.layer2(x)
        x =self.layer3(x)

        return x
#定义计算环境
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#数据集划分，加载为训练集验证集和测试集
custom_dataset =one_dataset("txt")
train_size = int(len(custom_dataset)*0.7)
val_size = int(len(custom_dataset)*0.2)
test_size = len(custom_dataset)-train_size-val_size
train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(custom_dataset,[train_size,val_size,test_size])

train_loader = DataLoader(train_dataset ,batch_size=16,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)
test_size = DataLoader(test_dataset,batch_size=1,shuffle=False)

print("训练集的大小",len(train_loader) *16,"验证集的大小",len(val_loader),"测试集的大小",len(test_loader))


def infer(model,dataset,device):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            datas,label =data
            outputs = model(datas.to(device))
            predict_y = torch.max(outputs,dim=1)[1]
            acc_num+= torch.eq(predict_y,label,to(device)).sum(),item()
    acc =acc_num/ len(dataset)
    return acc

def main(lr=0.005,epochs=20):
    model =NN(4,12,6,3).to(device)
    loss_f=nn.CrossEntropLoss()

    pg = [p for p in model.parameters()if p.requires_grad]
    optimizer = optim.Adam(pg,lr=lr)

    #权重文件存储路径
    save_path = os.path.join(os.getcwd(),"results/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    #开始训练
    for epoch in range(epochs):
        model.train()
        acc_num = torch.zeros(1).to(device)
        sample_num = 0

        train_bar = tqdm(train_loader,file=sys.stdout,ncols=100)
        for datas in train_bar:
            data,label = datas
            label = label.squeeze(-1)
            sample_num+=data.shape[0]

            optimizer.zero_grad()
            outputs = model(data.to(device))
            loss = loss_f(outputs,label.to(device))
            loss.backward()
            optimizer.step()

            train_acc = acc_num/sample_num
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch+1,epochs,loss,train_acc)
            val_acc = infer(model,val_loader,device)
            print("epoch[{}/{}] loss:{:.3f} acc:{:.3f} val_acc:{:.3f}".format(epoch+1,epochs,loss,train_acc,val_acc))
            torch.save(model.state_dict(),os.path.join(save_path,"nn.pth"))

            #每次数据集迭代后，要对初始化的指标清零
            train_acc=0.
            val_acc=0.
        print("Fished Training")

        test_acc = infer(model,test_loader,device)
        print("test_acc:{:.3f}".format(test_acc))

    if __name__ == "__main__":
        main()




