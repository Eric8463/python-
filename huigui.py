import numpy as np
import matplotlib.pyplot as plt

#读入训练数据
train=np.loadtxt('click.csv',delimiter=',',skiprows=1)
train_x=train[ : ,0]
train_y=train[ : ,1]

#绘图
plt.subplot(2,1,1)
plt.plot(train_x,train_y,'o')
plt.title('原始数据')
plt.xlabel('X')
plt.ylabel('y')
#plt.show()

#参数初始化
theta0=np.random.rand()
theta1=np.random.rand()

#预测函数，看数据分布预测的函数
def f(x):
    return theta0+theta1*x

#目标函数，该目标函数是二次函数，所以要找到最小值
def E(x,y):
    return 0.5 * np.sum((y - f(x)) ** 2)

#标准化
mu=train_x.mean()
sigma= train_x.std()
def standardize(x):
    return (x-mu)/sigma

train_z=standardize(train_x)
plt.subplot(2,1,2)
#plt.plot(train_z, train_y, 'o')
plt.title('标准化数据')
plt.xlabel('X')
plt.ylabel('y')
#plt.show()

#使用参数更新表达式去更新theta0/theta1，使得在训练数据下，目标函数值最小

#学习率，假设为10^-3
eta=1e-3
#误差的差值
diff=1
#更新次数
count=0

#重复学习
error = E(train_z,train_y)
while diff > 1e-2:
    #更新结果保存到临时变量
    tmp0 = theta0 - eta * np.sum((f(train_z) - train_y))
    tmp1 = theta1 - eta * np.sum((f(train_z) - train_y) * train_z)
    #更新参数
    theta0 = tmp0
    theta1 = tmp1
    #计算与上一次的误差
    current_error =E(train_z,train_y)
    diff =error - current_error
    error = current_error
    #输出日志
    count += 1
    log = '第{}次: theta0 ={:.3f},theta1 ={:.3f},差值 = {:.4f}'
    print(log.format(count,theta0,theta1,diff))
x = np.linspace(-3,3,100)
plt.plot(train_z,train_y,"o")
plt.plot(x,f(x))
plt.show()

#验证
print(f(standardize(100)))

