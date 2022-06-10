
#八、使用集成学习策略进行二分类
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
from sklearn.linear_model import LogisticRegression #线性回归
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.ensemble import RandomForestClassifier #随机森林
from sklearn.naive_bayes import GaussianNB #贝叶斯
from sklearn.svm import SVC #支持向量机SVM
from mlxtend.classifier import StackingCVClassifier
#一些评价模型性能的参数
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_curve,auc
import matplotlib.pyplot as plt
from time import time 
import datetime
model1  = LogisticRegression() 
model2 = DecisionTreeClassifier()
model3 = SVC()
meta_model = RandomForestClassifier()
time1 = time() 
JiCheng = StackingCVClassifier(classifiers=[model1,model2,model3], meta_classifier=meta_model, verbose = 1) #使用stacking集成学习
print("开始交叉验证：") 
y_pred1 = cross_val_predict(model1,x, y, cv=5) 
time2 = time() 
y_pred2 = cross_val_predict(model2,x, y, cv=5)  
time3 = time()
y_pred3 = cross_val_predict(model3,x, y, cv=2) 
time4 = time()
y_pred = cross_val_predict(JiCheng,x, y, cv=5)
time5 = time()

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
print("交叉验证所花时间为：",time5 - time4)
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

print("model1:")
cm1 = confusion_matrix(y, y_pred1)
accuracy1 = accuracy_score(y, y_pred1)
precision1 = precision_score(y, y_pred1)
recall1 = recall_score(y, y_pred1)
fpr1, tpr1, thersholds1 = roc_curve(y, y_pred1)
roc_auc1 = auc(fpr1, tpr1)
print("预测测试集“混淆矩阵”为：")
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)
disp1.plot()
plt.show()
print("交叉验证所花时间为：",time2 - time1)
print("预测测试集“准确率”为:",accuracy1)
print("预测测试集“精确率”为:",precision1)
print("预测测试集“召回率”为:",recall1)
print("预测测试集“ROC曲线”为：")
plt.plot(fpr1, tpr1, color='red', linestyle='-', label='ROC (area = {0:.3f})'.format(roc_auc1), lw=2)
plt.xlim([-0.05, 1.05])                #设置x、y轴的上下限，避免图像和边缘重合
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print("model2:")
cm2 = confusion_matrix(y, y_pred2)
accuracy2 = accuracy_score(y, y_pred2)
precision2 = precision_score(y, y_pred2)
recall2 = recall_score(y, y_pred2)
fpr2, tpr2, thersholds2 = roc_curve(y, y_pred2)
roc_auc2 = auc(fpr2, tpr2)
print("预测测试集“混淆矩阵”为：")
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot()
plt.show()
print("交叉验证所花时间为：",time3 - time2)
print("预测测试集“准确率”为:",accuracy2)
print("预测测试集“精确率”为:",precision2)
print("预测测试集“召回率”为:",recall2)
print("预测测试集“ROC曲线”为：")
plt.plot(fpr2, tpr2, color='red', linestyle='-', label='ROC (area = {0:.3f})'.format(roc_auc2), lw=2)
plt.xlim([-0.05, 1.05])                #设置x、y轴的上下限，避免图像和边缘重合
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print("model3:")
cm3 = confusion_matrix(y, y_pred3)
accuracy3 = accuracy_score(y, y_pred3)
precision3 = precision_score(y, y_pred3)
recall3 = recall_score(y, y_pred3)
fpr3, tpr3, thersholds3 = roc_curve(y, y_pred3)
roc_auc3 = auc(fpr3, tpr3)
print("预测测试集“混淆矩阵”为：")
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3)
disp3.plot()
plt.show()
print("交叉验证所花时间为：",time4 - time3)
print("预测测试集“准确率”为:",accuracy3)
print("预测测试集“精确率”为:",precision3)
print("预测测试集“召回率”为:",recall3)
print("预测测试集“ROC曲线”为：")
plt.plot(fpr3, tpr3, color='red', linestyle='-', label='ROC (area = {0:.3f})'.format(roc_auc3), lw=2)
plt.xlim([-0.05, 1.05])                #设置x、y轴的上下限，避免图像和边缘重合
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
