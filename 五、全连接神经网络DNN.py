
#五、使用全连接神经网络DNN进行二分类
#1.读取数据,划分x与y
import numpy as np         #大量数据函数
import pandas as pd        #读取csv，iloc
from sklearn.model_selection import train_test_split     #划分训练集和测试集
from sklearn.preprocessing import MinMaxScaler
#读取数据
data = pd.read_csv("pca_80_636.csv")
print("原始数据：",data.shape)
x = data.iloc[:,:-1]
y = data.iloc[:,[-1]]
y = y.values.ravel()
print("x为：",x.shape)
print("y为：",y.shape)
#标准化，否则难以收敛
scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

#2.构建模型
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from time import time
import datetime
time1 = time()
# 全连接神经网络
model = Sequential()
input = x.shape[1]#维度数
# 隐层，dense全连接层
model.add(Dense(units=64, input_shape=(input,), activation= 'relu'))   
model.add(Dropout(0.2))                        # Dropout层用于防止过拟合
# 隐层
model.add(Dense(units=64,activation= 'relu'))      #该层神经元数units，激活函数
model.add(Dropout(0.2))
# 隐层
model.add(Dense(units=64,activation= 'relu'))      #该层神经元数units，激活函数
model.add(Dropout(0.2))
# 输出层
model.add(Dense(units=1,activation= 'sigmoid'))
#print(model.summary())

#3.画层次图及训练预测
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_CNN.png', show_shapes=True, show_layer_names=False)
#loss损失函数；optimizer优化器；metrics评价函数
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])  
# early stoppping。monitor用于决定是否应终止训练；patience能够容忍多少个epoch内都没有improvement。
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=2)
# 训练。epochs迭代次数，verbose=0或1或2（不显示，显示，换行），batch_size每次梯度更新的样本数（默认32），callbacks回调函数
history = model.fit(x_train, y_train, epochs=500, verbose=2, batch_size=512, validation_data=(x_test, y_test), callbacks=[early_stopping])
# 预测
y_pre = model.predict_classes(x_test)
time2 = time()

#4.输出相关指标
#准确率，分类正确的样本数占总样本数的比例。
#精确率，预测为正类的样本中真正类所占的比例；越高，则模型对负样本区分能力越强。
#召回率，真实为正类的样本中被预测为正类的比例；越高，则模型对正样本的区分能力越强。
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_curve,auc

cm = confusion_matrix(y_test, y_pre)
accuracy = accuracy_score(y_test, y_pre)
precision = precision_score(y_test, y_pre)
recall = recall_score(y_test, y_pre)
fpr, tpr, thersholds = roc_curve(y_test, y_pre)
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
print("训练集和测试集“损失值”为：")
plt.plot(history.history['loss'], label='train')       #loss：训练集的损失值
plt.plot(history.history['val_loss'], label='test')    #val_loss：测试集的损失值
plt.xlabel('Epochs')
plt.ylabel('Loss Rate')
plt.title('Loss and Val_loss')
plt.legend()
plt.show()
print("训练集和测试集“准确率”为：")
plt.plot(history.history['accuracy'], label='train')       #accuracy：训练集的准确率
plt.plot(history.history['val_accuracy'], label='test')    #val_accuracy：测试集的准确率
plt.xlabel('Epochs')
plt.ylabel('Accuracy Rate') 
plt.title('Accuracy and Val_accuracy')
plt.legend() 
plt.show()

