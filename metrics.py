import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(true, pred, classes):
	N = len(classes)
	cm = np.zeros(shape=(N, N))
	for n in range(N):
		y, count = np.unique(pred[true == n], return_counts=True)
		for i in range(len(count)):
			cm[n][y[i]] = count[i]
	return cm

def precision_score(confusion_matrix, true_class):
	if confusion_matrix.shape[0] != 2:
		raise ValueError ("expected confusion matrix of shape 2x2")
	else:
		true_positives = confusion_matrix[true_class][true_class]
		wrong_positives = confusion_matrix[1 - true_class][true_class]
		if true_positives + wrong_positives == 0:
			return 0.0000001
		return (true_positives / (true_positives + wrong_positives))

def recall_score(confusion_matrix, true_class):
	if confusion_matrix.shape[0] != 2:
		raise ValueError ("expected confusion matrix of shape 2x2")
	else:
		true_positives = confusion_matrix[true_class][true_class]
		wrong_negatives = confusion_matrix[true_class][1 - true_class]
		if true_positives + wrong_negatives == 0:
			return 0.00000001
		return (true_positives / (true_positives + wrong_negatives))

def f1_score(confusion_matrix=None, true_class=None):
	if confusion_matrix.shape[0] != 2:
		raise ValueError ("expected confusion matrix of shape 2x2")
	else:
		precision = precision_score(confusion_matrix, true_class)
		recall = recall_score(confusion_matrix, true_class)
	return (2 * precision * recall / (precision + recall))

def compute_precision_recall_curve(y_pred_proba, y_true):
	thresholds = np.sort(np.unique(y_pred_proba))
	precision_scores = []
	recall_scores = []
	f1_scores = []
	for threshold in thresholds:
		y_pred = binarize_predictions(y_pred_proba, threshold)
		cm = confusion_matrix(y_true, y_pred, [0,1])
		precision_scores.append(precision_score(cm, 1))
		recall_scores.append(recall_score(cm, 1))
		f1_scores.append(f1_score(cm, 1))
	return [np.array(precision_scores), np.array(recall_scores), thresholds, f1_scores]

def plot_precision_recall_curve(prc_scores: dict):
	if len(prc_scores.values()) == 1:
		prc_score = prc_scores.values()[0]
		for i in range(0, len(prc_score[2]), 5):
			plt.annotate(f'{prc_score[2][i]:.2f}', (prc_score[1][i], prc_score[0][i]), textcoords="offset points", xytext=(0,10), ha='center')
	for k in prc_scores.keys():
		precision_scores, recall_scores, _, _ = prc_scores[k]
		plt.plot(recall_scores, precision_scores, marker='o', label=k)
		plt.xlabel("recall")
		plt.ylabel("precision")
	plt.legend()
	plt.show()

def plot_f1_threshold_curve(prc):
	plt.plot(prc[2], prc[3])
	plt.xlabel("threshold")
	plt.ylabel("f1-score")
	plt.show()

def binarize_predictions(y_pred, threshold):
	return (y_pred > threshold).astype(int)