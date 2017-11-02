#encoding=utf-8
#第10课：交叉验证过拟合


import numpy as np
from sklearn import datasets
from sklearn.learning_curve import validation_curve
from  sklearn.svm import SVC
import matplotlib.pyplot as plt

load_data=datasets.load_digits()
data_x=load_data.data
data_y=load_data.target
print len(data_x),len(data_y)
param_range=np.logspace(-6,-2.3,5)
train_loss,test_loss=validation_curve(SVC(),data_x,data_y,param_name='gamma',param_range=param_range,cv=10,scoring='mean_squared_error')

train_loss_mean=np.mean(-train_loss,axis=1)
test_loss_mean=np.mean(-test_loss,axis=1)

plt.plot(param_range,train_loss_mean,'o-',color='r',label='train_loss_mean')
plt.plot(param_range,test_loss_mean,'o-',color='g',label='test_loss_mean')
plt.xlabel('gamma')
plt.ylabel('loss')
plt.show()





