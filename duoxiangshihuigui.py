import numpy as np
import matplotlib.pyplot as plt

#读入训练数据
train=np.loadtxt('click.csv',delimiter=',',skiprows=1)
train_x=train[ : ,0]
train_y=train[ : ,1]
ETA = 1e-3
mu =train_x.mean()
sigma =train_x.std()
def standize(x):
    return (x-mu)/sigma
train_z=standize(train_x)
def E(x,y):
    return 0.5 * np.sum((y-f(x))**2)

#绘图
#plt.subplot(2,1,1)
#plt.plot(train_x,train_y,'o')
#plt.title('原始数据')
#plt.xlabel('X')
#plt.ylabel('y')
#plt.show()

#参数初始化
theta=np.random.rand(3)
#创建训练数据的矩阵
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]),x,x**2]).T
X = to_matrix(train_z)
#print(X)

#预测函数
def f(x):
    return np.dot(x,theta)
def MSE(x,y):
    return (1/(x.shape[0])*np.sum((y-f(x))**2))


#print(f(X))

#误差的差值
diff = 1

#重复学习
error = []
error.append(MSE(X,train_y))


while diff > 1e-2 :
    #更新参数
    theta = theta - ETA * np.dot(f(X) -train_y,X)
    #计算与上一次的误差
    error.append(MSE(X,train_y))
    diff = error[-2]-error[-1]

x=np.arange(len(error))
plt.plot(x, error)
plt.show()