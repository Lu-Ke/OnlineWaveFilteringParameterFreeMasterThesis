import numpy as np

from .optimizer import Optimizer

class Hedge(Optimizer):

	def __init__(self):
		self.theta = None
		self.eta = 1
	
	def iterate(self, grad):
		if(self.theta is None):
			self.theta = -grad
		else:
			self.theta -= grad

		result = np.exp(self.theta - 1) - 1

		norm = np.sum(result)
		if(norm >= 1e-6):
			# good case
			return result / np.sum(result)
		else:
			#bad case
			return result / (np.sum(result) + 1e-6)

	def clone(self):
		return Hedge()