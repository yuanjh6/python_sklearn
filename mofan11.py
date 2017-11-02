#encoding=utf-8
#第11课：模型保存和提取

from sklearn import svm
from sklearn import datasets
import pickle

clf=svm.SVC()
iris=datasets.load_iris()
x,y=iris.data,iris.target
clf.fit(x,y)

#save model
# with open('d:/clf.pickle','wb') as f:
#     pickle.dump(clf,f)

with open('d:/clf.pickle','rb') as f:
    clf2=pickle.load(f)
print clf2.predict(x[0:1]),y[0:1]