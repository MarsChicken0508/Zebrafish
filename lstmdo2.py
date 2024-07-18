import os

import numpy as np
import pandas as pd  #用來資料分析的
import torch         #深度學習用的
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler  #機器學習用的
import matplotlib.pyplot as plt     #資料庫用的
import plotly.express as px    #繪圖用的
import csv

def create_dataset(dataset,size):#分割測試集訓練集的副程式
    question,ans = [],[]
    for i in range(len(dataset) - size):
        q_array = dataset[i:i + size]
        a_array = dataset[i + 1:i + size + 1]
        ans.append(a_array)
        question.append(q_array)
    return torch.tensor(question, dtype=torch.float32), torch.tensor(ans, dtype=torch.float32)
    #return torch.tensor(question),torch.tensor(ans)

#LSTM模型
class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size= 2,hidden_size=64,num_layers=2,batch_first=True)#hidden_size = 64
        self.linear = nn.Linear(64,2)#(64,1)
        self.dropout = nn.Dropout(0.1)#0.1
    def forward(self,x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.linear(x)
        
        return x
   
def start_train(model,loader,optimizer,question_data,ans_data,train_loss_array,test_loss_array):
    for epoch in range(n_epochs):
        model.train()
        for question_batch, ans_batch in loader:
            question_batch = torch.randn(1, lookback, 2)
            ans_pred = model(question_batch)
            
            #let_me_look = ans_pred.view(0)
            #print(question_batch)
            #print("epoch=",epoch)
            #print(ans_pred)
            loss = loss_fn(ans_pred,ans_batch)
            '''
            ans_pred = 預測值
            ans_batch = 正確答案
            目前推測 可能是上面的lstm_model裡 forward中的dropout在linear等等有問題造成pred有很大的不同
            反正從forward去抓問題
            '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            ans_pred = model(question_data)
            train_loss = loss_fn(ans_pred,question_data[:, -1, :]).item()

            ans_pred = model(ans_data)
            test_loss = loss_fn(ans_pred,ans_data[:, -1, :]).item()
            #print(epoch)
            train_loss_array.append(train_loss)
            test_loss_array.append(test_loss)
        if epoch % 100 == 0:
            print("Epoch %d:train_loss %.4f, test_loss %.4f"%(epoch,train_loss,test_loss))
            
    with torch.no_grad():
        model.eval()

        train_plot = np.ones_like(question_data) * np.nan
        ans_pred = model(question_data)
        train_plot[lookback:train_size] = model(question_data)
        test_plot = np.ones_like(input) * np.nan
        test_plot[train_size + lookback : len(input)] = model(ans_data).numpy()
'''     train_loss_array.append(train_loss)
        test_loss_array.append(test_loss)'''
#-----------------------------------------------main--------------------------------------------------------
#讀檔
load = "data.csv"
read = pd.read_csv(load)
input = read[["x","y"]].values.astype('float32')

#切割訓練集and測試集大小
train_size = int(len(input) * 0.67)
test_size = len(input) - train_size
train_list,test_list = input[:train_size],input[train_size:]

lookback = 10#一次抓取測試資料大小

#切割好訓練集and測試集的數據以及各數據計算出的答案
question_train,ans_train = create_dataset(train_list,lookback)
question_test,ans_test = create_dataset(test_list,lookback)

model = lstm_model()
optimizer = optim.RMSprop(model.parameters())

loss_fn = nn.HuberLoss()
n_epochs = 101

loader = data.DataLoader(data.TensorDataset(question_train,ans_train),shuffle=False,batch_size=lookback)
train_loss_array = []
test_loss_array = []

start_train(model,loader,optimizer,question_train,question_test,train_loss_array,test_loss_array)

train_loss_array = []
test_loss_array = []

ans = model(question_test)
opt_loader = data.DataLoader(data.TensorDataset(question_test,ans_test),shuffle=False,batch_size=lookback)

# 開啟輸出的 CSV 檔案
with open('output.csv', 'w', newline='') as csvfile:# 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y'])
    for opt_array,ans_array in (opt_loader):
        #print(opt_array)
        opt_array = torch.randn(1, lookback, 2)
        with torch.no_grad():
            get_ans = model(opt_array)
            print(get_ans)
        writer.writerow([get_ans])