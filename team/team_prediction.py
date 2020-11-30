# coding=utf-8
import pandas as pd  # pandas是一个开源的，为Python编程语言提供高性能，易于使用的数据结构和数据分析工具。
import numpy as np  # NumPy是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
import xgboost as xgb  # 机器学习库
from sklearn.model_selection import train_test_split  # scikit-learn最大的特点就是，为用户提供各种机器学习算法接口，可以让用户简单、高效地进行数据挖掘和数据分析。
import shap  # SHAP is a unified approach to explain the output of any machine learning model.
import matplotlib.pyplot as pl  # matplotlib是 Python 2D-绘图领域使用最广泛的套件。
from sklearn.metrics import accuracy_score

shap.initjs()
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

prefix = "/Users/wzx/Documents/lol/"
teamstats = pd.read_csv(prefix+"ace.csv")


X = teamstats.drop(["win","matchid"], axis=1)
y = teamstats["win"]

Xt, Xv, yt, yv = train_test_split(X,y, test_size=0.2, random_state=10)
dt = xgb.DMatrix(Xt, label=yt.values)  # XGBoost加载训练数据
dv = xgb.DMatrix(Xv, label=yv.values)  # XGBoost加载测试数据

model = xgb.Booster({'nthread':4})  # init model
model.load_model("team.model")  # load data

preds = model.predict(dv)


predictions = [round(value) for value in preds]
y_test = dv.get_label()
test_accuracy = accuracy_score(y_test, predictions)
print(yv)
print(Xv)
print(preds)
print(predictions)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
