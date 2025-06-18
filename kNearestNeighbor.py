
from machine_learning import kdtree
from sklearn.model_selection import train_test_split

		
class KNN:
	def __init__(self, data, label):
		self.tree = kdtree.KDTree('l2')
		
	def fit(self, data, label):
		self.tree.fit(data, label)

### predict label of a point by getting the k closest neighbors ###
	def predict(self, q, k):
		result = []
		for point in q:
			self.tree.kNearestNeighbor(point, self.tree.root, k)
			null = 0
			for [node, _] in self.tree.kClosest:
				if node.label == 0:
					null += 1
			result = result + [0] if null > k // 2 else result + [1]
		return(result)
		
	def print_closest(self):
		for i, [node, dist] in enumerate(self.tree.kClosest):
			print(f"{i}-th closest point is {node.point} with a distance of {dist:.2f}")
