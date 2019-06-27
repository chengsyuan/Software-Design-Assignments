######################
# load dataset
######################
import pandas as pd

dataset = pd.read_csv('dataset.csv')

######################
# split dataset
######################
from sklearn.model_selection import train_test_split

train, test = train_test_split(dataset, test_size=10000, shuffle=True)

print('len(train) = {}, len(test) = {}'.format(len(train), len(test)))
print(train.columns)

x_cols = ['City', 'Gender', 'Age', 'Income']
y_cols = ['Illness']

train_x, train_y = train[x_cols], train[y_cols]
test_x, test_y = test[x_cols], test[y_cols]

######################
# feature engineering
######################

train_x_, test_x_ = train_x.to_dict('records'), test_x.to_dict('records')
train_y_, test_y_ = train_y.to_dict('records'), test_y.to_dict('records')

from sklearn.feature_extraction import DictVectorizer

x_vec = DictVectorizer()
train_x_array = x_vec.fit_transform(train_x_).toarray()
test_x_array = x_vec.transform(test_x_).toarray()
print('x_vec features:', x_vec.get_feature_names())

y_vec = DictVectorizer()
train_y_array = y_vec.fit_transform(train_y_).toarray()
test_y_array = y_vec.transform(test_y_).toarray()
print('y_vec features:', y_vec.get_feature_names())

######################
# MLP model
######################
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(train_x_array, train_y_array)
# print(clf.predict(test_x_array))

######################
# evaluate
######################
from sklearn.metrics import accuracy_score
print(accuracy_score(clf.predict(test_x_array),
                     test_y_array))