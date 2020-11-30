# coding=utf-8
import pandas as pd  # pandas是一个开源的，为Python编程语言提供高性能，易于使用的数据结构和数据分析工具。
import numpy as np  # NumPy是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
import xgboost as xgb  # 机器学习库
from sklearn.model_selection import train_test_split  # scikit-learn最大的特点就是，为用户提供各种机器学习算法接口，可以让用户简单、高效地进行数据挖掘和数据分析。
import shap  # SHAP is a unified approach to explain the output of any machine learning model.
import matplotlib.pyplot as pl  # matplotlib是 Python 2D-绘图领域使用最广泛的套件。

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

shap.initjs()

# 从csv中将数据读入Pandas的DataFrame
prefix = "/Users/wzx/Documents/lol/"
matches = pd.read_csv(prefix+"matches.csv")
participants = pd.read_csv(prefix+"participants.csv")
stats1 = pd.read_csv(prefix+"stats1.csv", low_memory=False)
stats2 = pd.read_csv(prefix+"stats2.csv", low_memory=False)
stats = pd.concat([stats1, stats2])

# 将众多Frame合并至一个Frame中
a = pd.merge(participants, matches, left_on="matchid", right_on="id")
allstats_orig = pd.merge(a, stats, left_on="id_x", right_on="id")
allstats = allstats_orig.copy()

# 筛选掉游戏时长小于10分钟的对局
allstats = allstats.loc[allstats["duration"] >= 10*60,:]

adc = allstats.loc[allstats["role"] == "DUO_CARRY"]
adc = adc.loc[adc["position"] == "BOT"]



X = adc.drop(["matchid","wardsbought","win","role","position","id_x","player","championid","ss1","ss2","role",
                   "id_y","gameid","platformid","queueid","seasonid","creation","version","id",
                   "item1","item2","item3","item4","item5","item6","trinket","killingsprees","dmgtoobj"
                   ,"legendarykills","magicdmgdealt","physicaldmgdealt","largestcrit","longesttimespentliving",
                   "truedmgdealt","physdmgtochamp","magicdmgtochamp","truedmgtochamp","totunitshealed",
                   "timecc","dmgselfmit","magicdmgtaken","physdmgtaken","truedmgtaken","goldspent","totcctimedealt","champlvl"
                   , "ownjunglekills","enemyjunglekills","firstblood","wardskilled","wardsplaced","pinksbought","visionscore"
                  ], axis=1)  # 删除win和wardsbought（无效数据）这一列
y = adc["win"]  # 将win这一列单独列出

X["totminionskilled"] = X["totminionskilled"]+X["neutralminionskilled"]
X = X.drop(["neutralminionskilled"],axis=1)
# 把所有数据按分钟来算平均值
rate_features = [
    "kills", "deaths", "assists", "doublekills",
    "triplekills", "quadrakills", "pentakills","largestkillingspree",
    "totdmgdealt",
    "totdmgtochamp",
    "totheal","totdmgtaken","dmgtoturrets",
     "goldearned",
    "totminionskilled",
]
for feature_name in rate_features:
    X[feature_name] /= (X["duration"] / 60)  # 除以分钟数，计算每分钟的数值

# 将最长存活时间除以游戏的总时长，计算比例
#X["longesttimespentliving"] /= X["duration"]

X = X.drop(["duration"],axis=1)

# 为属性定义一个更友好的名称
full_names = {
    "kills": "Kills per min.",
    "deaths": "Deaths per min.",
    "assists": "Assists per min.",


    "doublekills": "Double kills per min.",
    "triplekills": "Triple kills per min.",
    "quadrakills": "Quadra kills per min.",
    "pentakills": "Penta kills per min.",

    "totdmgdealt": "Total damage dealt per min.",

    "totdmgtochamp": "Total damage to champions per min.",

    "totheal": "Total healing per min.",



    "totdmgtaken": "Total damage taken per min.",

    "goldearned": "Gold earned per min.",

    "totminionskilled": "Total minions killed per min.",
    "turretkills": "# of turret kills",
    "inhibkills": "# of inhibitor kills",

    "dmgtoturrets": "Damage to turrets per min"
}
feature_names = [full_names.get(n, n) for n in X.columns]
X.columns = feature_names


# create train/validation split
# 划分训练子集和测试子集
# train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
# 格式：
# X_train,X_test, y_train, y_test =train_test_split(train_data,train_target,test_size=0.3, random_state=0)
# X：被划分的样本特征列
# y：被划分的样本标签
# test_size：在0-1之间，表示测试集占总样本的比例。
# random_state：是随机数的种子。
# Xt：训练集的特征列，Xv：测试集的特征列，yt：训练集的标签，yv：测试集的标签
Xt, Xv, yt, yv = train_test_split(X,y, test_size=0.2, random_state=10)
dt = xgb.DMatrix(Xt, label=yt.values)  # XGBoost加载训练数据
dv = xgb.DMatrix(Xv, label=yv.values)  # XGBoost加载测试数据


params = {
     "eta": 0.1,  # 如同学习率，向梯度方向前进的步长,缺省值为0.3
    "max_depth":15,  # 树的最大深度,缺省值为6
    "min_child_weight": 15,  # 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。
    "gamma": 0,  # 在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。
    "subsample": 0.8,  # 用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    "colsample_bytree": 0.2,  # 在建立树时对特征随机采样的比例。
    "scale_pos_weight": 1,  # 大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛
    "objective": "binary:logistic",  # 定义学习任务以及相应的学习目标，'binary:logistic'表示二分类的逻辑回归问题，输出为概率
    "silent": 0,   # silent:0表示打印出运行时信息，1表示以缄默方式运行，缺省值为0
    "base_score": np.mean(yt),  # 所有实例的初始预测分数。np.mean()方法是求平均值
    "eval_metric": "logloss",  # 校验数据所需要的评价指标，logless意思是负对数似然
}
# 利用xgb.train()方法训练模型，params是训练的参数，dt是用来训练的数据集，early_stopping_rounds=5意思是如果经过5轮训练之后
# 性能都没有提升，就会停止训练.[(dt, "train"),(dv, "test")]是Watchlist。Watchlist is used to specify validation set monitoring
# during training. For example user can specify watchlist=list(validation1=mat1, validation2=mat2) to watch the
# performance of each round's model on mat1 and mat2。verbose_eval=25意思是每隔25轮输出一次。
model = xgb.train(params, dt, 1000, [(dt, "train"), (dv, "test")], early_stopping_rounds=100,verbose_eval=25)

# 保存模型
model.save_model('adc.model')
