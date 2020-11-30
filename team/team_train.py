# coding=utf-8
import pandas as pd  # pandas是一个开源的，为Python编程语言提供高性能，易于使用的数据结构和数据分析工具。
import numpy as np  # NumPy是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
import xgboost as xgb  # 机器学习库
from sklearn.model_selection import train_test_split  # scikit-learn最大的特点就是，为用户提供各种机器学习算法接口，可以让用户简单、高效地进行数据挖掘和数据分析。
import shap  # SHAP is a unified approach to explain the output of any machine learning model.
import matplotlib.pyplot as pl  # matplotlib是 Python 2D-绘图领域使用最广泛的套件。

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

params = {
    "eta": 0.5,  # 如同学习率，向梯度方向前进的步长,缺省值为0.3
    "max_depth": 4,  # 树的最大深度,缺省值为6
    "objective": "binary:logistic",  # 定义学习任务以及相应的学习目标，'binary:logistic'表示二分类的逻辑回归问题，输出为概率
    "silent": 1,   # silent:0表示打印出运行时信息，1表示以缄默方式运行，缺省值为0
    "base_score": np.mean(yt),  # 所有实例的初始预测分数。np.mean()方法是求平均值
    "eval_metric": "logloss"  # 校验数据所需要的评价指标，logless意思是负对数似然
}
# 利用xgb.train()方法训练模型，params是训练的参数，dt是用来训练的数据集，early_stopping_rounds=5意思是如果经过5轮训练之后
# 性能都没有提升，就会停止训练.[(dt, "train"),(dv, "test")]是Watchlist。Watchlist is used to specify validation set monitoring
# during training. For example user can specify watchlist=list(validation1=mat1, validation2=mat2) to watch the
# performance of each round's model on mat1 and mat2。verbose_eval=25意思是每隔25轮输出一次。
model = xgb.train(params, dt, 300, [(dt, "train"), (dv, "test")], early_stopping_rounds=5, verbose_eval=25)

model.save_model('team.model')


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xv)


shap.summary_plot(shap_values, Xv)