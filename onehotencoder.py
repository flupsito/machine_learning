import numpy as np

class OneHotEncoder():

	def __init__(self, data):
		self.data = data

	def fit(self) -> dict:
		uni = np.unique(self.data)
		self.label_enc = dict(zip(uni, np.arange(0, len(uni), 1)))
		self.label_dec = dict(zip(np.arange(0, len(uni), 1), uni))

	def encode(self):
		self.arr = np.zeros([np.shape(self.data)[0], len(self.label_enc)])
		i = 0
		for d in self.data:
			l = self.label_enc[d]
			self.arr[i][l] = 1
			i = i + 1
		return np.vstack(self.arr)

	def decode(self):
		self.rev_arr = []
		j = 0
		for i in self.arr:
			self.rev_arr.append(self.label_dec[np.argwhere(i == 1)[0][0]])
		return np.hstack(self.rev_arr)

	
if __name__ == "__main__":
	data = np.genfromtxt("bearing_faults.csv",dtype="str", delimiter=",")
	Encoder = OneHotEncoder(data=data)
	Encoder.fit()
	ohe_values = Encoder.encode()
	print(f'one-hot encoded representation: \n {ohe_values}')
	dec_values = Encoder.decode()
	a = data
	b = dec_values
	res = all(x == y for x, y in zip(a, b))
	if res:
		print("true encoding and decoding")
	else:
		print("WRONG")
