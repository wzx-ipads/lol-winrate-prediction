import pandas as pd
import xgboost as xgb
import sys


def predict(argument):
    model = xgb.Booster({'nthread': 4})  # init model
    model.load_model("/Users/wzx/python-workspace/lol2/support/support.model")  # load data

    test = pd.DataFrame({'Kills per min.': [argument[0]],
                         'Deaths per min.': [argument[1]],
                         'Assists per min.': [argument[2]],
                         'largestmultikill': [argument[3]],

                         'Double kills per min.': [argument[4]],
                         'Triple kills per min.': [argument[5]],
                         'Quadra kills per min.': [argument[6]],
                         'Penta kills per min.': [argument[7]],

                         'Total damage dealt per min.': [argument[8]],

                         'Total damage to champions per min.': [argument[9]],
                         'Total healing per min.': [argument[10]],

                         'Total damage taken per min.': [argument[11]],
                         'visionscore per min': [argument[12]],
                         'Gold earned per min.': [argument[13]],
                         'Total minions killed per min.': [argument[14]],
                         'Pink wards bought per min.': [argument[15]],
                         'Wards placed per min.': [argument[16]],
                         'wards killed per min.': [argument[17]]
                         })

    dtest = xgb.DMatrix(test)
    preds = model.predict(dtest)

    print preds[0]


if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a.append((float(sys.argv[i])))

    predict(a)