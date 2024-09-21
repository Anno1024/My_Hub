import numpy as np
from matplotlib import pyplot as plt


class sinx_model():
    def __init__(self):
        self.epochs = 500000
        self.lr = 0.00005
        self.scalar = 1
        self.interval = 10000
        self.x,self.y = self.data_init(100)
        self.paras = np.random.rand(6)
        self.data = self.data_loader(self.x,self.y)

    #准备数据进行拟合
    def data_init(self, n):
        x = np.linspace(-np.pi, np.pi, n)
        y = np.sin(x)
        return x, y

    def data_loader(self,x,y):
        list = []
        for i in range(len(x)):
            list.append(np.array([pow(x[i],5),pow(x[i],4),pow(x[i],3),pow(x[i],2),pow(x[i],1), 1.0]))
        np_array = np.vstack(list)
        return np_array

    def Loss(self, x, y):
        loss = []
        for i in range(len(x)) :
            loss_value = (self.paras[0]*pow(x[i],5)+self.paras[1]*pow(x[i],4)+self.paras[2]*pow(x[i],3)+self.paras[3]*pow(x[i],2)+self.paras[4]*pow(x[i],1)+ self.paras[5]-y[i])**2
            loss.append(loss_value)
        loss = np.array(loss)
        return np.mean(loss)

    # 使用单条训练数据进行梯度下降非常不友好,咱们不用这个
    def train(self):
        for i in range(self.epochs):
            count = 0
            for i in range(len(self.data)):

                error =(self.data[i] @self.paras.T) -self.y[i]
                w_grad = error*self.paras
                self.paras = self.paras - self.lr*w_grad
                if count % self.interval == 0:
                    print(self.Loss(self.x,self.y))

    # 进行梯度更新
    def grad_convey(self,x,error):
        error = error.reshape(-1, 1)
        result = x*error
        result = result.T
        grad_w = []
        for i in result:
            grad_w.append(np.mean(i))
        grad_w = np.array(grad_w)
        self.paras = self.paras - self.lr*grad_w




    # 整体数据进行梯度下降，相当于一个大的batch
    def train2(self):
        for i in range(self.epochs):
            x = self.data
            output = x@self.paras.T
            error = output - self.y
            self.grad_convey(x,error)

            if i % self.interval == 0:
                print(self.Loss(self.x,self.y))




    def plt_draw(self):
        plt.title("sinx fit")
        plt.scatter(self.x,self.y)

        y_predict = []
        for i in range(len(self.x)) :
            y_predict.append([self.paras[0]*pow(self.x[i],5)
                              +self.paras[1]*pow(self.x[i],4)
                              +self.paras[2]*pow(self.x[i],3)
                              +self.paras[3]*pow(self.x[i],2)
                              +self.paras[4]*pow(self.x[i],1)
                              +self.paras[5]])
        y_predict = np.array(y_predict)

        plt.plot(self.x,y_predict,'r')
        plt.show()
k = sinx_model()
k.train2()
loss = k.Loss(k.x,k.y)
k.plt_draw()
print(k.paras)
print(loss)
