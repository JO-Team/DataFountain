import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
import scipy as sp

train = pd.read_csv('../input/user_action_train.txt', sep="\t", header=None, parse_dates=[3])
goods = pd.read_csv('../input/goods_train.txt', sep="\t", header=None)
test = pd.read_csv('../input/user_action_test_items.txt.', sep="\t", header=None)

# remove useless column of user_action_text_item.txt
test = test.drop(test.columns[[2]], axis=1)

train.rename(columns={0: 'uid', 1: 'spu_id', 2: 'action_type', 3: 'date'}, inplace=True)
goods.rename(columns={0: 'spu_id_goods', 1: 'brand_id', .2: 'cat_id'}, inplace=True)
test.rename(columns={0: 'uid', 1: 'spu_id'}, inplace=True)

train_goods = pd.merge(train, goods, left_on='spu_id', right_on='spu_id_goods', how='left')
train_goods.rename(columns={5: 'cat_id'}, inplace=True)

train_label = train['action_type']
train_goods = train_goods.drop(['action_type', 'date', 'spu_id_goods', 'brand_id'], axis=1)
train_goods = train_goods.drop([2], axis=1)
print(train_goods.head())

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(train_goods, train_label)
dtest = xgb.DMatrix(test)

num_boost_rounds = 100

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=num_boost_rounds, verbose_eval=50, show_stdv=False)
print(cv_output[['train-rmse-mean', 'test-rmse-mean']])

print(len(cv_output))
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_predict = model.predict(dtest)
model_output = pd.DataFrame({'weight': y_predict})
print(model_output)

model_output.to_csv('submisstion_wu.txt', index=False)
