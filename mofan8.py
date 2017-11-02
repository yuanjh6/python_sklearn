#encoding=utf-8
#第8课：交叉验证

from  sklearn import datasets

load_data=datasets.load_iris()
data_x=load_data.data
data_y=load_data.target

from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

k_range=range(1,31)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(k)
    score=-cross_val_score(knn,data_x,data_y,cv=10,scoring='mean_squared_error')
    # score=cross_val_score(knn,data_x,data_y,cv=10,scoring='accuracy')
    k_scores.append(score.mean())
plt.plot(k_range,k_scores)
plt.xlabel('k value for knn')
plt.ylabel('mean score for knn')
plt.show()
