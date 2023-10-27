import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#设置数据路径
data_path = os.path.abspath("C://Users//yhh68//Downloads//archive")
#创建存储图像和标签的列表
images=[]#存储图像数据的列表
labels= []#存储标签数据的列表

#遍历每个标签文件夹并收集数据
for label_folder in os.listdir(data_path):
    folder_path=os.path.join(data_path, label_folder)
    
    #如果不是文件夹,则跳过
    if not os.path.isdir(folder_path):
        continue
    
    #遍历文件夹内的JPG,png图像并收集数据
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (32, 32))
            image = image.flatten()
            images.append(image)
            labels.append(label_folder)

#转换为NumPy数组
X=np.array(images)#图像数据
y=np.array(labels)#标签数据

#数据集分割
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#创建基础决策树分类器
base_estimator=DecisionTreeClassifier(max_depth=2)

#创建AdaBoost分类器并进行训练
adaboost=AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)

#使用测试数据进行预测
y_pred=adaboost.predict(X_test)

#计算准确度
accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy:{accuracy * 100:.2f}%")

#将测试准确度保存到列表中
test_accuracies=list(adaboost.staged_score(X_test,y_test))

#可视化测试准确度
plt.figure(figsize=(8,6))
plt.plot(range(1,51),test_accuracies, marker='o',linestyle='-')
plt.title('Test Accuracy Over Boosting Iterations')
plt.xlabel('Number of Boosting Iterations')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.show()
