import numpy as np
import pandas as pd

train = pd.read_csv('../input/user_action_train.txt', sep="\t", header=None, parse_dates=[3])
goods = pd.read_csv('../input/goods_train.txt', sep="\t", header=None)
test = pd.read_csv('../input/user_action_test_items.txt.', sep="\t", header=None)

# remove useless column of user_action_text_item.txt
test = test.drop(test.columns[[2]], axis=1)

train.rename(columns={0: 'uid', 1: 'spu_id', 2: 'action_type', 3: 'date'}, inplace=True)
goods.rename(columns={0: 'spu_id_goods', 1: 'brand_id', .2: 'cat_id'}, inplace=True)
test.rename(columns={0: 'uid', 1: 'spu_id'}, inplace=True)

print("user_action_train.txt: ")
print(train.head())
print("goods_train.txt: ")
print(goods.head())
print("user_action_test_items.txt: ")
print(test.head())