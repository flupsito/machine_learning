import numpy as np
import matplotlib.pyplot as plt
import Zscore as zs
from sklearn.cluster import DBSCAN as DBSCANNER
from sklearn.metrics import silhouette_score
from collections import deque
import argparse
import time
from sklearn.neighbors import KDTree

class dbscan:

	def __init__(self, eps=1, N_min=2):
		self.data: np.ndarray
		self.cores: deque
		self.next_cluster = 0
		self.eps = eps
		self.N_min = N_min
		self.tree: KDTree

	def dbscan(self):
		zscore = zs.Zscorer()
		data = np.loadtxt("data_clustering.csv", delimiter=',')
		zscore.fit(data)
		data = zscore.transorm(data)
		self.tree = KDTree(data)
		self.data = np.full(shape=(data.shape[0], 5), fill_value=-1, dtype=np.float64)
		self.data[:,0:3] = np.array(data, dtype=np.float64)
		self.cores = deque()

	def plot(self):
		if self.next_cluster > 7:
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			s = ax.scatter(self.data[:,0], self.data[:,1], self.data[:,2], c=self.data[:,3], cmap="plasma")
			plt.colorbar(s)
		else:
			fig, subs = plt.subplots(3,3)
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
		
	def eucl_3d_dist(self, x, y):
		if x.shape != y.shape:
			raise ValueError ("both points are from differant range")
		return np.sqrt(np.sum(x - y) ** 2)
	
	def sum_of_close_points(self, candidate):
		sum_close_points = 1
		for point in self.data:
			if self.eucl_3d_dist(point, candidate) < self.eps:
				sum_close_points = sum_close_points + 1
			else:
				continue
		return sum_close_points
	
	def is_core(self, i, candidate):
		N = len(self.tree.query_radius([candidate[0:3]], r=self.eps)[0])
		if  N >= self.N_min:
			candidate[4] = 1
			self.cores.append(i)
		return
	
	def get_core_candidates(self):
		for i, candidate in enumerate(self.data):
			self.is_core(i, candidate)
	
	def assign_cluster(self):
		while self.cores:
			core_idx = self.cores.popleft()
			core = self.data[core_idx]
			if core[3] != -1:
				continue
			else:
				core[3] = self.next_cluster
				self.next_cluster = self.next_cluster + 1
				self.add_neighbors(core)
				
	def add_neighbors(self, core):
		deepseek = deque()
		deepseek.append(core)
		while deepseek:
			current = deepseek.popleft()
			for candidate_idx in self.cores:
				candidate = self.data[candidate_idx]
				if candidate[3] != -1:
					continue
				elif self.eucl_3d_dist(current, candidate) < self.eps:
					candidate[3] = core[3]
					deepseek.append(candidate)
				else:
					continue
	
	def get_edges(self):
		for candidate in self.data:
			if candidate[4]!=-1:
				continue
			else:
				for core in self.data[self.data[:,4] == 1]:
					if self.eucl_3d_dist(candidate, core) < self.eps:
						candidate[3] = core[3]
					else:
						continue		

def main():
	start_time = time.time()
	scanner = dbscan(eps=0.5, N_min=15)
	scanner.dbscan()
	scanner.get_core_candidates()
	scanner.assign_cluster()
	scanner.get_edges()
	end_time = time.time()
	print(f"calculation of clusters took {end_time - start_time} seconds")
	print(f"in the dataset there are: {scanner.next_cluster} cluster")
	if scanner.next_cluster > 0:
		score = silhouette_score(scanner.data[:,0:3], scanner.data[:,3])
		print(f"silhouette_score = {score}")
	scanner.plot()

if __name__ == "__main__":
	main()
	
	
	

	