import pandas as pd
import xgboost as xgb
import sys


def predict(argument):
    
    model = xgb.Booster({'nthread': 4})  # init model
    model.load_model("/Users/wzx/python-workspace/lol2/jungle/jungle.model")  # load data

    test = pd.DataFrame({'Kills per min.': [argument[0]],
                         'Deaths per min.': [argument[1]],
                         'Assists per min.': [argument[2]],
                         'largestmultikill': [argument[3]],
                         'largest killingspree per min': [argument[4]],

                         'Double kills per min.': [argument[5]],
                         'Triple kills per min.': [argument[6]],
                         'Quadra kills per min.': [argument[7]],
                         'Penta kills per min.': [argument[8]],

                         'Total damage dealt per min.': [argument[9]],

                         'Total damage to champions per min.': [argument[10]],

                         'Damage to objects per min.': [argument[11]],
                         'Damage to turrets per min': [argument[12]],
                         'Total damage taken per min.': [argument[13]],
                         'visionscore per min': [argument[14]],
                         'Gold eirned per min.': [argument[15]],
                         'Total minions killed per min.': [argument[16]],
                         'Neutral minions killed per min.': [argument[17]],
                         'Own jungle kills per min.': [argument[18]],
                         'Enemy jungle kills per min.': [argument[19]],
                         'Pink wards bought per min.': [argument[20]],
                         '# of turret kills': [argument[21]],
                         '# of inhibitor kills': [argument[22]],
                         'Wards placed per min.': [argument[23]],

                         'first blood': [argument[24]]})

    dtest = xgb.DMatrix(test)
    preds = model.predict(dtest)

    print preds[0]


if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a.append((float(sys.argv[i])))

    predict(a)





