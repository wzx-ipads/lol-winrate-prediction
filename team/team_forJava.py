import pandas as pd
import xgboost as xgb
import sys


def predict(argument):
    model = xgb.Booster({'nthread': 4})  # init model
    model.load_model("/Users/wzx/python-workspace/lol2/team/team.model")  # load data

    test = pd.DataFrame({'firstblood': [argument[0]],
                         'firsttower': [argument[1]],
                         'firstinhibit.': [argument[2]],
                         'firstbaron': [argument[3]],
                         'firstdragon': [argument[4]],

                         'firstharry': [argument[5]],
                         'towerkills1': [argument[6]],
                         'towerkills2': [argument[7]],
                         'inhibkill1': [argument[8]],

                         'inhibkills2': [argument[9]],

                         'baronkills1': [argument[10]],

                         'baronkills2': [argument[11]],
                         'dragonkills1': [argument[12]],
                         'dragonkills2.': [argument[13]],
                         'kills1': [argument[14]],
                         'kills2': [argument[15]],
                         'deaths1': [argument[16]],
                         'deaths2': [argument[17]],
                         'assists1': [argument[18]],
                         'assists2': [argument[19]],
                         'goldearned1': [argument[20]],
                         'goldearned2': [argument[21]],
                         'dmg1': [argument[22]],
                         'dmg2': [argument[23]]})

    dtest = xgb.DMatrix(test)
    preds = model.predict(dtest)

    print preds[0]


if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a.append((float(sys.argv[i])))

    predict(a)

