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

def create_dataset(dataset,size,xy):#分割測試集訓練集的副程式
    question,ans = [],[]
    for i in range(len(dataset) - size):
        q_array = dataset[i:i + size][xy]
        a_array = dataset[i + 1:i + size + 1][xy]
        ans.append(a_array)
        question.append(q_array)
    return torch.tensor(question),torch.tensor(ans)

#LSTM模型
class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size= 2,hidden_size=64,num_layers=2,batch_first=True)#hidden_size = 64
        self.linear = nn.Linear(64,1)#(64,1)
        self.dropout = nn.Dropout(0.1)#0.1
    def forward(self,x):
        #print(x)
        x, _ = self.lstm(x)
        #print(x)
        x = self.dropout(x)
        #print(x)
        x = self.linear(x)
        #print(x)
        return x


def start_train(model,loader,optimizer,question_data,ans_data,train_loss_array,test_loss_array):
    for epoch in range(n_epochs):
        #print(epoch)
        model.train()
        for question_batch, ans_batch in loader:
            ans_pred = model(question_batch)
            #print(ans_pred)
            #print(ans_batch)
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
            train_loss = loss_fn(ans_pred,question_data).item()

            ans_pred = model(ans_data)

            test_loss = loss_fn(ans_pred,ans_data).item()
            #print(epoch)
            train_loss_array.append(train_loss)
            test_loss_array.append(test_loss)
        if epoch % 100 == 0:
            print("Epoch %d:train_loss %.4f, test_loss %.4f"%(epoch,train_loss,test_loss))
            
    with torch.no_grad():
        model.eval()

        train_plot = np.ones_like(input[ :, xy]) * np.nan
        ans_pred = model(question_data)
        train_plot[lookback:train_size] = model(question_data).view(-1)
        test_plot = np.ones_like(input[ :, xy]) * np.nan
        test_plot[train_size + lookback : len(input)] = model(ans_data).view(-1)
    '''
    plt.plot(input[xy])
    plt.plot(train_plot)
    plt.plot(test_plot)
    plt.show()
    print(train_plot)
    print(test_plot)
    '''
    #blue = ans,orange = train_pred,green = test_pred

#-----------------------------------------------main--------------------------------------------------------
#讀檔
load = "F:\\Desktop\\Graduation Topic\\data.csv"
read = pd.read_csv(load)
input = read[["x","y"]].values.astype('float32')

#切割訓練集and測試集大小
train_size = int(len(input) * 0.67)
test_size = len(input) - train_size
train_list,test_list = input[:train_size],input[train_size:]

lookback = 15#一次抓取測試資料大小

#切割好訓練集and測試集的數據以及各數據計算出的答案
x_question_train,x_ans_train = create_dataset(train_list,lookback,0)
y_question_train,y_ans_train = create_dataset(train_list,lookback,1)
x_question_test,x_ans_test = create_dataset(test_list,lookback,0)
y_question_test,y_ans_test = create_dataset(test_list,lookback,1)

'''
print(x_question_train)
print(x_ans_train)
print(y_question_train)
print(y_ans_train)
print(x_question_test)
print(x_ans_test)
print(y_question_test)
print(y_ans_test)

print(x_question_train.size,x_ans_train.size)
print(y_question_train.size,y_ans_train.size)
print(x_question_test.size,x_ans_test.size)
print(y_question_test.size,y_ans_test.size)
'''

model_x = lstm_model()
model_y = lstm_model()
optimizer_x = optim.RMSprop(model_x.parameters())
optimizer_y = optim.RMSprop(model_y.parameters())
loss_fn = nn.HuberLoss()
n_epochs = 1000


loader_x = data.DataLoader(data.TensorDataset(x_question_train,x_ans_train),shuffle=False,batch_size=lookback)
loader_y = data.DataLoader(data.TensorDataset(y_question_train,y_ans_train),shuffle=False,batch_size=lookback)
train_loss_array = []
test_loss_array = []
xy = 0
start_train(model_x,loader_x,optimizer_x,x_question_train,x_question_test,train_loss_array,test_loss_array)

#start_train(model_x,loader_x,optimizer_x,x_question_train,x_ans_train,train_loss_array,test_loss_array)
train_loss_array = []
test_loss_array = []
xy = 1
start_train(model_y,loader_y,optimizer_y,y_question_train,y_question_test,train_loss_array,test_loss_array)
get_ans = model_x(train_list)
print(get_ans)
#    for i in range(len(train_total_plot)):
#        writer.writerow([str(train_total_plot[i][0]),str(train_total_plot[i][1])])
    