#数据加载操作必须复写父类的三个方法
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
import  torch

class onedataloader(Dataset):
     def __init__(self,data_path):
         self.data_path=data_path

         assert os.path.exists(self.data_path), "'datasets does not exists"

         df=pd.read_csv(self.data_path,name=[0,1,2,3,4])

         d={"iris-set-one":0,"iris-set-two":1}

         df[4] =df[4].map(d)

         data =df.iloc[:,0:4]
         label=df.iloc[:,4: ]

        data=(data-np.mean(data)/np.std(data))

        self.data= torch.form_numpy(np.array(data,dtype=np.float32))
        self.label=torch.from_numpy(np.array(label,dtype=np.int64))

        self.data_num=len(lable)
        print("当前数据集大小为："，self.data_num)

     def __len__(self):
         return self.data_num

     def __getitem__(self,index):
         self.data=list(self.data)

         self.label=list(self.lable)

         return self.data[index],self.label[index]










