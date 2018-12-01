#encoding=utf-8
#第9课：过拟合

import numpy as np
from sklearn import datasets
from sklearn.learning_curve import learning_curve
from  sklearn.svm import SVC
import matplotlib.pyplot as plt

load_data=datasets.load_digits()
data_x=load_data.data
data_y=load_data.target
print len(data_x),len(data_y)
train_sizes,train_loss,test_loss=learning_curve(SVC(gamma=0.001),data_x,data_y,cv=10,scoring='mean_squared_error',train_sizes=np.linspace(0.05,1,10))

train_loss_mean=np.mean(-train_loss,axis=1)
test_loss_mean=np.mean(-test_loss,axis=1)

plt.plot(train_sizes,train_loss_mean,'o-',color='r',label='train_loss_mean')
plt.plot(train_sizes,test_loss_mean,'o-',color='g',label='test_loss_mean')
plt.xlabel('train_sizes')
plt.ylabel('loss')
plt.show()


