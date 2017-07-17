# *coding=utf-8*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
import scipy as sp
#import missingno as msno
from scipy.stats import mode

train = pd.read_table('../input/user_action_train.txt', sep="\t", header=None)
goods = pd.read_table('../input/goods_train.txt', sep="\t", header=None)
test = pd.read_table('../input/user_action_test_items.txt', sep="\t", header=None)

test = test.drop(test.columns[[2]], axis=1)

#train[3] = pd.to_datetime(train[3])
train.rename(columns={0:'uid',1:'spu_id',2:'action_type',3:'date'},inplace=True)
goods.rename(columns={0:'spu_id_goods',1:'brand_id',2:'cat_id'},inplace=True)
test.rename(columns={0:'uid',1:'spu_id'},inplace=True)
print(pd.merge(train,goods,left_on='spu_id',right_on='spu_id_goods',how='left'))

#将训练购买记录和商品表连接并删除一些useless
train_goods = pd.merge(train,goods,left_on='spu_id',right_on='spu_id_goods',how='left')

#55594876个0,578010个1，共56172886个训练数据，5761092个测试数据
print('显示一下1和0的数量')
group = train_goods.groupby(train_goods['action_type'])
group_sum = group.count()
print(group_sum)

train_0 = train_goods[train_goods.action_type==0]
print(np.shape(train_0))
train_1 = train_goods[train_goods.action_type==1]
print(np.shape(train_1))


train_split = pd.concat([train_0.sample(n=580000*5),train_1])
np.random.shuffle(np.array(train_split))

print(np.shape(train_split))

#获取训练label
train_label = train_split['action_type']

dftrain = train_split.drop(['action_type','date','spu_id_goods'],axis=1)



#将测试购买记录和商品表连接并删除一些useless
test_goods = pd.merge(test,goods,left_on='spu_id',right_on='spu_id_goods',how='left')
dftest = test_goods.drop(['spu_id_goods'],axis=1)


# print(train_goods)
# print(train_label)
# print(test_goods)





xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

#train_label = np.log(train_label)
dtrain = xgb.DMatrix(dftrain, label=train_label)
dtest = xgb.DMatrix(dftest)

# cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,verbose_eval=20, show_stdv=False)
# cv_output[['train-rmse-mean', 'test-rmse-mean']]

num_boost_rounds = 460 #len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_predict = model.predict(dtest)
model_output = pd.DataFrame({'weight': y_predict})
model_output.to_csv('result1_5.csv',index=False)
print(model_output)

