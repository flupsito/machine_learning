import numpy as np
import matplotlib.pyplot as plt

def _split(self, x, y, k):
	N = x.shape[0]
	splits_x = list()
	splits_y = list()
	size = N // k
	for i in range(k):
		splits_x.append(x[size * i: size * (i + 1)])
		splits_y.append(y[size * i: size * (i + 1)])
	return splits_x, splits_y
	
def _accuracy(self, y_pred, y_true):
	return np.mean(y_pred == y_true)

def k_fold_split(x, y, k, seed=None):
	splits_x, splits_y = _split(x, y, k)
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	for i in range(k):
		x_test.append(splits_x[i])
		y_test.append(splits_y[i])
		x_train.append(np.concatenate(splits_x[:i] + splits_x[i + 1:]))
		y_train.append(np.concatenate(splits_y[:i] + splits_y[i + 1:]))
	return x_train, x_test, y_train, y_test

def evaluate_cv(model, x_train, y_train, x_test, y_test):
	acc = []
	while x_train:
		x_tr = x_train.pop()
		y_tr = y_train.pop()
		x_te = x_test.pop()
		y_te = y_test.pop()
		model.fit(x_tr, y_tr)
		pred = model.predict(x_te)
		acc.append(_accuracy(pred, y_te))
	return np.mean(acc), np.std(acc)
