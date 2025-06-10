import numpy as np
import water_data as wd
from sortedcontainers import SortedKeyList
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class Node:
	def __init__(self, point, axis, label, left_child = None, right_child = None):
		self.point = point
		self.label = label
		self.axis = axis
		self.left_child = left_child
		self.right_child = right_child

class KDTree:
	def __init__(self, metric):
		self.root = None
		if metric not in ['l2', 'se2', 'angles']:
			raise ValueError ("invalid metric")
		if metric == 'l2':
			self.metric = l2
		elif metric == 'se2':
			self.metric = se2
		else:
			self.metric = angle
		self.kClosest = SortedKeyList(key=lambda x: -x[1])
		self.rClose = []
		self.axis: int

### fit data in KD-Tree ###
	def fit(self, q, label):
		self.axis = q.shape[1]
		self.point_list = []
		self.getInsertOrder(q, label)
		self.addConfigIterative()

	def getInsertOrder(self, q, label, depth = 0):
		if len(q) == 0:
			return
		axis = depth % self.axis
		sort_idx = q[:,axis].argsort()
		q_sort = q[sort_idx]
		label_sort = label[sort_idx]
		median = len(q) // 2
		self.point_list.append([q_sort[median], label_sort[median]])
		self.getInsertOrder(q_sort[:median], label_sort[:median], depth + 1)
		self.getInsertOrder(q_sort[median + 1:], label_sort[median + 1:], depth + 1)

	def addConfigIterative(self):
		if self.point_list is None:
			return
		for [point, label] in self.point_list:
			self.insert(point, label)
	
	def insert(self, point, label):
		if self.root is None:
			self.root = Node(point, 0, label)
			return
		curr = self.root
		depth = 0
		count = 0
		while True:
			axis = depth % self.axis
			if curr.point[axis] > point[axis]:
				if curr.left_child is None:
					curr.left_child = Node(point, (depth + 1) % self.axis, label)
					return
				curr = curr.left_child
			else:
				if curr.right_child is None:
					curr.right_child = Node(point, (depth + 1) % self.axis, label)
					return
				curr = curr.right_child
			depth += 1
		

### predict label of a point by getting the k closest neighbors ###
	def predict(self, q, k):
		result = []
		for point in q:
			self.kClosest = SortedKeyList(key=lambda x: -x[1])
			self.nearestKRecursive(point, self.root, k)
			null = 0
			for [node, _] in self.kClosest:
				if node.label == 0:
					null += 1
			result = result + [0] if null > k // 2 else result + [1]
		return(result)
		
	def nearestKRecursive(self, q, node: Node, k):
		if not node:
			return
		distance = self.metric(q, node.point)
		if len(self.kClosest) < k:
			self.kClosest.add([node, distance])
		elif self.kClosest[0][1] > distance:
				self.kClosest.pop(0)
				self.kClosest.add([node, distance])
		hyperplane_dist = q[node.axis] - node.point[node.axis]
		if  hyperplane_dist <= 0:
			close =  node.left_child
			far = node.right_child
		else:
			far = node.left_child
			close = node.right_child
		self.nearestKRecursive(q, close, k)
		if len(self.kClosest) < k or np.abs(hyperplane_dist) < self.kClosest[0][1]:
			self.nearestKRecursive(q, far, k)

	def print_closest(self):
		for i, [node, dist] in enumerate(self.kClosest):
			print(f"{i}-th closest point is {node.point} with a distance of {dist:.2f}")

### return everyting in a radius r ###
	def nearestR(self, q, r):
		self.rClose = []
		self.nearestRRecursive(q, r, self.root)

	def nearestRRecursive(self, q, r, node: Node):
		if not node:
			return
		if self.metric(node.point, q) < r:
			self.rClose.append(node.point)
		if not node.left_child  or not node.right_child:
			if not node.left_child and node.right_child:
				return self.nearestRRecursive(q, r, node.right_child)
			elif not node.right_child:
				return self.nearestRRecursive(q, r, node.left_child)
			return node.point
		elif q[node.axis] - node.point[node.axis] <= 0:
			close =  self.nearestRRecursive(q, r, node.left_child)
			far = self.nearestRRecursive(q, r, node.right_child)
		else:
			far =  self.nearestRRecursive(q, r, node.left_child)
			close = self.nearestRRecursive(q, r, node.right_child)
		self.nearestRRecursive(q, r, close)
		if q[node.axis] - node.point[node.axis] < self.rClose[0][node.axis]:
			self.nearestRRecursive(q, r, far)

### metric functions ###
def l2(point1, point2):
	return np.sqrt(np.sum((point1 - point2) ** 2))

def angle(angle1, angle2):
	dist = 0
	for i in range(len(angle1)):
		dist1 = np.abs(angle1[i] - angle2[i])
		if dist1 > np.pi:
			dist1 = 2 * np.pi - dist1
		dist += dist1
	return dist

def se2(point1, point2):
	return (l2(point1[:2], point2[:2]) + angle(point1[2], point2[2]))



def main():
	metric = 'l2'
	x, y = wd.load_water_data()
	X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)
	tree = KDTree(metric=metric)
	tree.fit(X_train, y_train)
	for k in range(1, 12, 2):
		pred = tree.predict(X_test, k)
		wrong = (pred == y_test)[(pred == y_test) == False]
		print(f"own impl:\tin total {len(wrong)} - {((len(wrong) / len(x)) * 100):.2f}% wrong predictions with k = {k}.")

if __name__ == '__main__':
	main()