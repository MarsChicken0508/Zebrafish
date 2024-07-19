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
        #hidden_size=128, num_layers=3, dropout=0.2
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
            #question_batch = torch.randn(1, lookback, 2)
            ans_pred = model(question_batch)
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
        if epoch % 10 == 0:
            print("Epoch %d:train_loss %.4f, test_loss %.4f"%(epoch,train_loss,test_loss))
def output(model,loader,optimizer,question_data):
    model.train()
    for question_batch, ans_batch in loader:
        ans_pred = model(question_batch)
        print(ans_pred)
        #optimizer.zero_grad()
        #optimizer.step()

    model.eval()
    with torch.no_grad():
        ans_pred = model(question_data)
#        ans_pred = model(ans_data)
#-----------------------------------------------main--------------------------------------------------------
#讀檔
load = "data.csv"
read = pd.read_csv(load)
input = read[["x","y"]].values.astype('float32')

#切割訓練集and測試集大小
train_size = int(len(input) * 0.67)
test_size = len(input) - train_size
train_list,test_list = input[:train_size],input[train_size:]

lookback = 5#一次抓取測試資料大小

#切割好訓練集and測試集的數據以及各數據計算出的答案
question_train,ans_train = create_dataset(train_list,lookback)
question_test,ans_test = create_dataset(test_list,lookback)
print(ans_test)
model = lstm_model()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

loss_fn = nn.HuberLoss()
n_epochs = 650
loader = data.DataLoader(data.TensorDataset(question_train,ans_train),shuffle=False,batch_size=lookback)
train_loss_array = []
test_loss_array = []
start_train(model,loader,optimizer,question_train,question_test,train_loss_array,test_loss_array)

train_loss_array = []
test_loss_array = []

ans = model(question_test)
opt_loader = data.DataLoader(data.TensorDataset(question_test,ans_test),shuffle=False,batch_size=lookback)
# 训练模型
#start_train(model, loader, optimizer, question_train, question_test, train_loss_array, test_loss_array)
#output(model,loader,optimizer,question_test)
# 保存最佳模型
torch.save(model.state_dict(), 'best_lstm_model.pth')
print("Model saved as 'best_lstm_model.pth'")


# 開啟輸出的 CSV 檔案
with open('output.csv', 'w', newline='') as csvfile:# 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y'])
    for opt_array,ans_array in (opt_loader):
        model.train()
        with torch.no_grad():
            get_ans = model(opt_array)
            x = 0
            y = 0
            for i in get_ans:
                x += i.tolist()[0]
                y += i.tolist()[1]
            x /= lookback
            y /= lookback
            x = int(x)
            y = int(y)
            print(x,y)
        writer.writerow([x,y])