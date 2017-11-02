#encoding=utf-8
#第5课：数据集介绍
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

# load_data=datasets.load_boston()
# data_x=load_data.data
# data_y=load_data.target
# train_x,test_x,train_y,test_y=train_test_split(data_x,data_y,test_size=0.3)
# lr=LinearRegression()
# lr.fit(train_x,train_y)
# print lr.predict(test_x)-test_y

x,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
plt.scatter(x,y)
plt.show()



