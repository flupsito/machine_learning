import numpy as np
from matplotlib import pyplot as plt


class Node:
    def __init__(self,
                 split_dim: int = None, 
                 split_val: float = None,
                 child_left=None, child_right=None,
                 prediction: int = None):
        self.split_dim = split_dim  
        self.split_val = split_val
        self.child_left = child_left
        self.child_right = child_right
        self.prediction = prediction
        
    def is_leaf(self):
        return self.prediction is not None
    
def majority_vote(y: np.ndarray) -> int:
    y_uni, counts = np.unique(y, return_counts=True)
    majority_class = y_uni[np.argmax(counts)]
    return majority_class

def entropy(y: np.ndarray) -> float:
	N = len(y)
	n_l = len(y[y==1])
	n_r = len(y[y==0])
	if n_l == 0 or n_r == 0:
		return 0
	H = - n_l / N * np.log2(n_l / N) - n_r / N * np.log2(n_r / N)
	return H

def information_gain(y_parent: np.ndarray, index_split: np.ndarray) -> float:
	N = len(index_split)
	n_l = len(index_split[index_split==1])
	n_r = len(index_split[index_split==0])
	if n_l == 0 or n_r == 0:
		return 0
	H_p = entropy(y_parent)
	H_l = n_l / N * entropy(y_parent[index_split[:,0] == 1])
	H_r = n_r / N * entropy(y_parent[index_split[:,0] == 0])
	info_gain = H_p - H_l - H_r
	return info_gain

def create_split(x: np.ndarray, j: int, split_val: float):
	index_split = np.expand_dims(x[:,j] <= split_val, axis=1)
	return index_split

def best_split(x: np.ndarray, y: np.ndarray):
	max_gain = 0
	split_dim = 0
	split_val = 0.0
	for j in range(x.shape[1]):
		for i in range(x.shape[0]):
			index_split = create_split(x, j, x[i][j])
			gain = information_gain(y, index_split)
			if gain > max_gain:
				max_gain = gain
				split_dim = j
				split_val = x[i][j]
	return split_dim, split_val, max_gain

class DecisionTree:
	def __init__(self, max_depth: int = 5, min_samples: int = 2):
		self.root = None 
		self.max_depth = max_depth
		self.min_samples = min_samples
		self.class_labels = None
		self.n_samples = None
		self.n_features = None
		self._curr_no_samples: int = None
		self.is_completed: bool = None
		self._curr_node_pure: bool = None

	def terminate(self, depth):
		if depth >= self.max_depth:
			return True
		elif self._curr_no_samples < self.min_samples:
			return True
		elif self._curr_node_pure:
			return True
		return False

	def _grow_tree(self, x: np.ndarray, y: np.ndarray, curr_depth: int = 0):
		node = Node()
		self._curr_node_pure = len(np.unique(y)) == 1
		self._curr_no_samples = x.shape[0]
		if self.terminate(curr_depth):
			node.prediction = majority_vote(y)
			return node
		else:
			split_dim, split_val, _ = best_split(x, y)
			node.split_dim = split_dim
			node.split_val = split_val
			node.child_left = self._grow_tree(x[x[:,split_dim] <= split_val], y[x[:,split_dim] <= split_val], curr_depth + 1)
			node.child_right = self._grow_tree(x[x[:,split_dim] > split_val], y[x[:,split_dim] > split_val], curr_depth + 1)
		return node

	def _traverse_tree(self, x: np.ndarray, node: Node = None) -> int:
		if node.is_leaf():
			return node.prediction
		elif x[node.split_dim] <= node.split_val:
			return self._traverse_tree(x, node.child_left)
		else:
			return self._traverse_tree(x, node.child_right)

	def fit(self, x: np.ndarray, y: np.ndarray):
		self.class_labels = np.unique(y)
		self.n_samples = x.shape[0]
		self.n_features = x.shape[1]
		self.root = self._grow_tree(x, y)

	def predict(self, x: np.ndarray):
		predictions = list()
		for sample in x:
			predictions.append(self._traverse_tree(sample, self.root))
		return np.array(predictions)

if __name__ == '__main__':
	data = np.loadtxt('decision_tree_dataset.txt', delimiter=',')
	x_train = data[:, :2]
	y_train = data[:, -1].astype(int)
	DT = DecisionTree(10,3)
	DT.fit(x=x_train, y=y_train)
	x_val = x_train
	y_val = y_train
	y_pred = DT.predict(x_val)
	print(f'\n\nground truth labels: \t {y_val}')
	print(f'predicted labels: \t \t {y_pred}')
