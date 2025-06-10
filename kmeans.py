from matplotlib import pyplot as plt
import numpy as np
import sys

def main():
	k = 4
	data = np.genfromtxt("example_data_Kmeans.csv", delimiter=",")
	centroids = get_centroids(k, data)
	
	# plt.scatter(data[:,0], data[:,1])
	# plt.show()
	old_c = np.zeros((k, 3))
	while not_converged(centroids, old_c):
		d = data_allocation(centroids, data)
		new_centr = update_centroids(d, k, centroids)
		old_c = centroids
		centroids = new_centr
		print(f"current sse:\n{calc_SSE(centroids,d,-1)}")
		print(f"current bss:\n{calc_bss(centroids, d)}")
	print(f"final centroids:\n{centroids}")
	print(calc_SSE(centroids,d,-1))
	plot_data(d, centroids)
	return d, centroids
	

def plot_data(d, centroids):
	s = plt.scatter(d[:,0], d[:,1], c=d[:,2], cmap="plasma")
	index = np.array([0, 1, 2, 3])
	plt.scatter(centroids[:,0], centroids[:,1], c=index, cmap="plasma",marker="x")
	plt.colorbar(s)
	plt.show()

def get_centroids(k, data):
	x = data[:,0]
	y = data[:,1]

	x_min = np.min(x)
	x_max = np.max(x) - x_min
	y_min = np.min(y)
	y_max = np.max(y) - y_min
	centroids = np.ndarray(shape=(k,3))
	for i in range(k):
		centroids[i][0] = x_min + x_max * np.random.uniform(0, 1)
		centroids[i][1] = y_min + y_max * np.random.uniform(0, 1)
		centroids[i][2] = i
	return centroids

def l2_dist(centroid, value) -> int:
	distmin = 10000
	index = 0
	for center in centroid:
		dist = np.sqrt((center[0] - value[0]) ** 2 + (center[1] - value[1]) ** 2)
		if dist < distmin:
			distmin = dist
			index = center[2]
	return index

def l1_dist(centroid, value) -> int:
	distmin = 100000
	index = 0
	for center in centroid:
		dist = np.abs(center[0] - value[0]) + np.abs(center[1] - value[1])
		if dist < distmin:
			distmin = dist
			index = center[2]
	return index

def data_allocation(centroids, data):
	a = np.ndarray(shape=((100,3)))
	for k, d in enumerate(data):
		i = l2_dist(centroids, d)
		a[k][0] = d[0]
		a[k][1] = d[1]
		a[k][2] = int(i)
	return a

def update_centroids(alloc_data, k, centr):
	centroids = np.zeros(shape=(k, 3))
	empty = -1
	for i in range(k):
		if len(alloc_data[:,0][alloc_data[:,2]==i]) == 0:
			empty = i
			centroids[i][2] = i
		else:
			centroids[i][0] = np.mean(alloc_data[:,0], where=alloc_data[0:,2]==i)
			centroids[i][1] = np.mean(alloc_data[:,1], where=alloc_data[0:,2]==i)
			centroids[i][2] = i
	if empty > -1:
		sse = calc_SSE(centroids, alloc_data, empty)
		centroids[empty][0], centroids[empty][1] = relocate_centroid(sse, alloc_data)
	return centroids

def not_converged(old_c, new_c) -> bool:
	if old_c.shape != new_c.shape:
		raise ValueError(f"Shape mismatch: old_c has shape {old_c.shape}, but new_c has shape {new_c.shape}")
	for k, c in enumerate(old_c):
		if (c[0] - new_c[k][0] > 0.05):
			return True
		elif (c[1] - new_c[k][1] > 0.05):
			return True
		else:
			continue
	return False

def check_empty_centroid(centroid) -> bool:
	if np.isnan(centroid[0]) or np.isnan(centroid[1]):
		return True
	else:
		return False
	
def calc_SSE(centroids, data, empty):
	sse = np.zeros(shape=(4,1))
	for k, center in enumerate(centroids):
		if k == empty:
			continue
		for d in data:
			if d[2] == k:
				sse[k] = sse[k] + (center[0] - d[0]) ** 2 + (center[1] - d[1]) ** 2
			else:
				continue
	return sse

def calc_bss(centroids, data):
	mean_x = np.mean(data[:,0])
	mean_y = np.mean(data[:,1])
	bss = 0
	for center in centroids:
		bss = bss + (center[0] - mean_x) ** 2 + (center[1] - mean_y) ** 2
	return bss

def relocate_centroid(sse, data):
	i = np.argmax(sse)
	x_min = data[:,0][data[:,2]==i].min()
	x_max = data[:,0][data[:,2]==i].max() - x_min
	y_min = data[:,1][data[:,2]==i].min()
	y_max = data[:,1][data[:,2]==i].max() - y_min
	centroid_x = x_min + x_max * np.random.uniform(0, 1)
	centroid_y = y_min + y_max * np.random.uniform(0, 1)
	return centroid_x, centroid_y

def return_cluster(centroids, new_entry) -> int:
	min_dist = 1000000
	index = -1
	for i, center in enumerate(centroids):
		dist = np.sqrt((center[0] - new_entry[0]) ** 2 + (center[1] - new_entry[1]) ** 2)
		if  dist < min_dist:
			index = i
			min_dist = dist
		else:
			continue
	return index
		
def update_mean(centroid, new_value, N):
	new_centroid = np.ndarray(shape=(1,3))
	new_centroid[0][0] = (N * centroid[0] + new_value[0]) / (N + 1)
	new_centroid[0][1] = (N * centroid[1] + new_value[1]) / (N + 1)
	new_centroid[0][2] = centroid[2]
	return new_centroid

def add_new_datapoint(data, new_value, centroids):
	i = return_cluster(centroids, new_value)
	data = np.vstack([data, [new_value[0], new_value[1], i]])
	N = len(data[:,0][data[:,2]==i])
	centroids[i] = update_mean(centroids[i], new_value, N)
	return data, centroids

if __name__ == "__main__":
	data, centroids = main()
	while True:
		new_entry = sys.stdin.readline()
		new_entry = new_entry.strip()
		if str(new_entry) == "exit":
			break
		else:
			new_value = new_entry.partition(',')
			new_value = [float(new_value[0]), float(new_value[2])]
			print(new_value)
			data, centroids = add_new_datapoint(data, new_value, centroids)
			plot_data(data,centroids)


