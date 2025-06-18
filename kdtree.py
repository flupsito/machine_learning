import numpy as np
from sortedcontainers import SortedKeyList

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
		self.kClosest = None
		self.rClose = None
		self.axis: int

### fit data in KD-Tree ###
	def fit(self, data, label = None):
		self.axis = data.shape[1]
		self.point_list = []
		if label == None:
			label = np.zeros(shape=(data.shape[0], 1))
		self.getInsertOrder(data, label)
		self.addConfigIterative()

### create a list in which order the nodes need to be inserted	###
### work around to overcome recursion depth limit of python   	###
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
### inserts the nodes in the precalculated order				###
	def insert(self, point, label):
		if self.root is None:
			self.root = Node(point, 0, label)
			return
		curr = self.root
		depth = 0
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

### handles a new request of finding k-nearest neighbors		###
	def kNearestNeighbor(self, q, node: Node, k):
		self.kClosest = SortedKeyList(key=lambda x: -x[1])
		self.kNearestNeighborRecursive(q, node, k)

### recursivly scans the kdTree for the k nearest neighbors to	###
### a given point n												###
	def kNearestNeighborRecursive(self, q, node: Node, k):
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
		self.kNearestNeighborRecursive(q, close, k)
		if len(self.kClosest) < k or np.abs(hyperplane_dist) < self.kClosest[0][1]:
			self.kNearestNeighborRecursive(q, far, k)

### handles a new request of finding all neighbors in radius r	###
	def rNearestNeighbor(self, q, r):
		self.rClose = SortedKeyList(key=lambda x: x[1])
		self.rNearestNeighborRecursive(q, r, self.root)

### recursivly scans the kdTree for all nearest neighbors in	###
### radius r to a given point q
	def rNearestNeighborRecursive(self, q, r, node: Node):
		if not node:
			return
		dist = self.metric(node.point, q)
		if  dist < r:
			self.rClose.add([node.point, dist])
		hyperplane_dist = q[node.axis] - node.point[node.axis]
		if hyperplane_dist <= 0:
			close =  node.left_child
			far = node.right_child
		else:
			far =  node.left_child
			close = node.right_child
		self.rNearestNeighborRecursive(q, r, close)
		if hyperplane_dist < self.rClose[0][0][node.axis]:
			self.rNearestNeighborRecursive(q, r, far)

### metric functions ###
def l2(point1, point2):
	if point1.shape != point2.shape:
		raise ValueError ("the shape of both points is not matching")
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
	