import pandas as pd

actions = pd.read_csv('../input/user_action_train.txt', sep="\t", header=None)
goods = pd.read_csv('../input/goods_train.txt', sep="\t", header=None)
test = pd.read_csv('../input/user_action_test_items.txt.', sep="\t", header=None)

print("user_action_train.txt: ")
print(actions.head())
print("goods_train.txt: ")
print(goods.head())
print("user_action_test_items.txt: ")
print(test.head())

# remove useless column of user_action_text_item.txt
test = test.drop(test.columns[[2]], axis=1)
print("user_action_test_items.txt: ")
print(test.head())