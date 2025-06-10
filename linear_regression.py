import numpy as np

#############################
#####	exercise a	#########
#############################

def lin_regression(x: float, y: float) -> float:
	n = np.shape(x)[0]
	a = np.array([[n, np.sum(x)], [np.sum(x), np.sum(x*x)]])
	b = np.array([[np.sum(y)], [np.sum(y*x)]])
	a_inv = np.linalg.inv(a)
	theta = np.matmul(a_inv, b)
	return theta

#############################
#####	exercise b	#########
#############################

data = np.genfromtxt("driving_data.csv", delimiter=",")

def rolling_resistance(data):
	C_w = 0.4
	A = 1.5		#m^2
	P_air = 1.2	#kg/m^3
	G = 9.81	#m/s^2
	M = 2400	#kg
	v = data[:,0]
	p = data[:,1]
	F_wind = C_w * A * P_air * (v ** 2) * 0.5
	p_wind = v * F_wind
	p_roll = p - p_wind
	theta = lin_regression(v * M * G, p_roll)
	x = v * M * G
	y = p_roll
	r2 = r2_dist(theta[0] + x * theta[1], y)
	return theta, r2

####################
#### exercise c ####
####################


def r2_dist(x: np.ndarray, y: np.ndarray) ->float:
	if type(x) != np.ndarray and type(y) != np.ndarray:
		raise TypeError("one of the inputs is unequal ndarray data type")
	if np.shape(x) != np.shape(y):
		raise ValueError("dimensions are not matching")
	y_mean = np.mean(y)
	numerator = np.sum((y - x) ** 2)
	denumerator = np.sum((y - y_mean) ** 2)
	if denumerator == 0:
		return 0
	else:
		return 1 - (numerator / denumerator)
	