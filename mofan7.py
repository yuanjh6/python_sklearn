#encoding=utf-8
#第7课：标准化数据

from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.svm    import SVC

x,y=make_classification(n_samples=300,n_features=2,n_informative=2,n_clusters_per_class=1,n_redundant=0,random_state=22,scale=100)
# plt.scatter(x[:,0],x[:,1],c=y)
# plt.show()

#x=preprocessing.scale(x)
#x=preprocessing.maxabs_scale(x)
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3)
svc=SVC()
svc.fit(train_x,train_y)
print svc.score(test_x,test_y)

