import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from collections import deque
from machine_learning import kdtree
from machine_learning import zscore as zs

class dbscan:

	def __init__(self, eps=1, N_min=10):
		self.data: np.ndarray
		self.cores = deque()
		self.next_cluster = 0
		self.eps = eps
		self.N_min = N_min
		self.tree = kdtree.KDTree('l2')

	def dbscan(self, data):
		self.dim = data.shape[1]
		zscore = zs.Zscorer()
		zscore.fit(data)
		data = zscore.transorm(data)
		self.tree.fit(data)
		self.data = np.full(shape=(data.shape[0], self.dim + 2), fill_value=-1, dtype=np.float64)
		self.data[:,0:self.dim] = np.array(data, dtype=np.float64)
		print(self.data.shape)

	def plot(self):
		if self.next_cluster > 7:
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			s = ax.scatter(self.data[:,0], self.data[:,1], self.data[:,2], c=self.data[:,3], cmap="plasma")
			plt.colorbar(s)
		else:
			fig, _ = plt.subplots(3,3)
			for i in range(-1,self.next_cluster + 2):
				if i == self.next_cluster + 1:
					s = ax.scatter(self.data[:,0], self.data[:,1], self.data[:,2], c=self.data[:,3], cmap="plasma")
					ax.set_xlim(-2,2)
					ax.set_ylim(-2,2)
					ax.set_zlim(-2,2)
					plt.colorbar(s)
				else:
					ax = fig.add_subplot(3,3,i+2,projection='3d')
					ax.scatter(self.data[:,0][self.data[:,3]==i], self.data[:,1][self.data[:,3]==i], self.data[:,2][self.data[:,3]==i])
					ax.set_xlim(-2,2)
					ax.set_ylim(-2,2)
					ax.set_zlim(-2,2)
		plt.savefig("plot.png")
		plt.show()
	
	def is_core(self, i):
		self.tree.rNearestNeighbor(self.data[i][0:self.dim], self.eps)
		N = len(self.tree.rClose)
		if  N >= self.N_min:
			self.data[i][self.dim + 1] = 1
			self.cores.append(i)
	
	def get_core_candidates(self):
		for i, _ in enumerate(self.data):
			self.is_core(i)
	
	def assign_cluster(self):
		while self.cores:
			core_idx = self.cores.popleft()
			if self.data[core_idx][self.dim] != -1:
				continue
			else:
				self.data[core_idx][self.dim] = self.next_cluster
				self.next_cluster = self.next_cluster + 1
				self.add_neighbors(core_idx)
				
	def add_neighbors(self, core_idx):
		deepseek = deque()
		deepseek.append(core_idx)
		while deepseek:
			current_idx = deepseek.popleft()
			for candidate_idx in self.cores:
				candidate = self.data[candidate_idx]
				if candidate[self.dim] != -1:
					continue
				elif self.tree.metric(self.data[current_idx][0:self.dim], candidate[0:self.dim]) < self.eps:
					candidate[self.dim] = self.data[core_idx][self.dim]
					deepseek.append(candidate_idx)
	
	def get_edges(self):
		candidates = self.data[self.data[:,self.dim + 1] != -1]
		for candidate in candidates:
			for core in self.data[self.data[:,self.dim + 1] == 1]:
				if self.tree.metric(candidate, core) < self.eps:
					candidate[self.dim] = core[self.dim]	

	def fit(self):
		self.get_core_candidates()
		self.assign_cluster()
		self.get_edges()
	
	

	