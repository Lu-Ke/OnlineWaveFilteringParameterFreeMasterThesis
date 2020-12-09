import numpy as np

from .optimizer import Optimizer

class FTRL(Optimizer):

	def __init__(self):
		self.theta = None
		self.eta = 1
		self.theta_square_sum = None
	
	def iterate(self, grad):
		if(self.theta is None):
			self.theta = -grad
		else:
			self.theta -= grad

		if(self.theta_square_sum is None):
			self.theta_square_sum = np.linalg.norm(grad) ** 2
		else:
			self.theta_square_sum += np.linalg.norm(grad) ** 2

		#return self.eta * self.theta / (np.sqrt(self.theta_square_sum) + 1e-6)#np.where(self.theta_square_sum <= 1e-6, 0.0, self.eta * self.theta / (np.sqrt(self.theta_square_sum) + 1e-6))
		return np.where(self.theta_square_sum <= 1e-6, 0.0, self.eta * self.theta / (np.sqrt(self.theta_square_sum) + 1e-6))

	def clone(self):
		return FTRL()