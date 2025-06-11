import numpy as np
import matplotlib.pyplot as plt


class Zscorer:
	def __init__(self):
		self.mean: float
		self.std_dev: float


	def fit(self, x: np.ndarray):
		self.mean = np.mean(x, axis=0)
		self.std_dev = np.std(x, axis=0)

	def transorm(self, x: np.ndarray):
		x_z = 1 / self.std_dev * (x - self.mean)
		return x_z
	
	def inverse_transform(self, x_z):
		x = x_z * self.std_dev + self.mean
		return x

	def plot(self, data):
		for x in data:
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			ax.scatter(x[:,0], x[:,1], x[:,2])
			ax.set_title("original data")
			plt.show()
		

if __name__ == "__main__":
	data = np.loadtxt("data_clustering.csv", delimiter=",")
	data = np.array(data, dtype=np.float64)
	zscore = Zscorer()
	zscore.fit(data)
	data_z = zscore.transorm(data)
	data_inv = zscore.inverse_transform(data_z)
	print(f'original data: \n {data[:3]} \n')
	print(f'transformed data: \n {data_z[:3]} \n')
	print(f'inverse transform: \n {data_inv[:3]}')
	print(f"mean = {zscore.mean}, std_dev = {zscore.std_dev}")
	zscore.plot([data, data_z, data_inv])