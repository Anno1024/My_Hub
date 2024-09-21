from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

class MyDataset():
    def __init__(self, csv_path):
        data = read_csv(csv_path)
        #df = data[['RM','PTRATIO','LSTAT','MEDV']]
        # todo: you can chose the column to see the different feature
        df = data[['RM',  'MEDV']]
        df_cleaned = df.dropna()
        self.data = df_cleaned.iloc[:,:-1].to_numpy(dtype=float)

        # self.means = self.data.mean(axis=0)
        # self.stds = self.data.std(axis=0)
        # self.data = (self.data - self.means) / self.stds

        self.fistcolumn = df_cleaned[['RM']]
        self.target = df_cleaned.iloc[:,-1].to_numpy(dtype=float)



    def __len__(self):
        print(len(self.data),len(self.target))
        return len(self.target)

    def __getitem__(self, idx):
        return self.data[idx],self.target[idx]

class Train():

    def __init__(self, Dataset,data_demention):
        self.Dataset = Dataset
        self.b = np.ones(1,dtype=float)
        self.w = np.random.rand(data_demention)
        self.lr = 0.00009
        self.scaler = 1
        self.interval = 10000
        self.epochs = 1000000

    #最小二乘法来定义loss
    def Loss_function(self,x,y):
        # 最小二乘法
        y_pred = (x@((self.w).T)) +self.b
        error = y_pred - y
        loss = np.mean(np.square(error))
        return loss

    def model(self,x):
        y_pred = (x@((self.w).T)) + self.b
        return y_pred

    def back_ward_function(self,x,error):
        b_grad = (1/len(x)) *(error.sum())
        w_grad = (1/len(x)) * (x.T@error)
        # 更新参数
        self.w = self.w - self.lr * w_grad
        self.b = self.b - self.lr * b_grad

    def gradient_descent(self):
        epochs = self.epochs
        count = 0
        for epoch in range(epochs):
            x = self.Dataset.data
            y = self.Dataset.target
            predict = self.model(x)
            error = predict - y
            self.back_ward_function(x,error)
            if count % self.interval == 0:
                self.lr = self.lr*self.scaler
                loss = self.Loss_function(x, y)
                print(loss)
            count +=1

data = MyDataset('HousingData.csv')
train = Train(data,data_demention=1)
train.gradient_descent()
print(train.w[0],train.b)


plt.title("house_price_predict")
plt.xlabel("RM")
plt.ylabel("price")
plt.scatter(data.fistcolumn,data.target)

y_predict = train.w[0]*data.fistcolumn+train.b
plt.plot(data.fistcolumn,y_predict,'r')
plt.show()




