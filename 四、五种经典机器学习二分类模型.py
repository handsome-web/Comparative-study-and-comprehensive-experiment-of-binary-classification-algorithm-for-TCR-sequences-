
#四、使用五种经典机器学习进行二分类
#1.读取数据,划分x与y
import numpy as np         #大量数据函数
import pandas as pd        #读取csv，iloc
from sklearn.model_selection import train_test_split #划分训练集和测试集
#读取数据
data = pd.read_csv("pca_80_636.csv")
print("原始数据：",data.shape)
x = data.iloc[:,:-1]
y = data.iloc[:,[-1]]
y = y.values.ravel()
print("x为：",x.shape)
print("y为：",y.shape)

#2.交叉验证即模型训练
#五种经典二分类
from sklearn.linear_model import LogisticRegression        #线性回归
from sklearn.tree import DecisionTreeClassifier            #决策树
from sklearn.svm import SVC                                #支持向量机SVM
from sklearn.neighbors import KNeighborsClassifier         #KNN
from sklearn.naive_bayes import GaussianNB                 #贝叶斯
from sklearn.ensemble import RandomForestClassifier #随机森林
#一些评价模型性能的参数
from sklearn.model_selection import cross_val_score,cross_val_predict 
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_curve,auc
import matplotlib.pyplot as plt
from time import time
import datetime

#五种经典二分类  
#model = LogisticRegression()
#model = DecisionTreeClassifier()
#model = SVC()
#model = KNeighborsClassifier(n_neighbors=3)
#model = GaussianNB()
time1 = time()
print("开始交叉验证：")
y_pred = cross_val_predict(model,x, y, cv=5)
#print(y_pred)
print("交叉验证结束！")
time2 = time()

#3.输出相关指标
#准确率，分类正确的样本数占总样本数的比例。
#精确率，预测为正类的样本中真正类所占的比例；越高，则模型对负样本区分能力越强。
#召回率，真实为正类的样本中被预测为正类的比例；越高，则模型对正样本的区分能力越强。
cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
fpr, tpr, thersholds = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)
print("预测测试集“混淆矩阵”为：")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
print("交叉验证所花时间为：",time2 - time1)
print("预测测试集“准确率”为:",accuracy)
print("预测测试集“精确率”为:",precision)
print("预测测试集“召回率”为:",recall)
print("预测测试集“ROC曲线”为：")
plt.plot(fpr, tpr, color='red', linestyle='-', label='ROC (area = {0:.3f})'.format(roc_auc), lw=2)
plt.xlim([-0.05, 1.05])                #设置x、y轴的上下限，避免图像和边缘重合
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

