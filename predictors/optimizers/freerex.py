import numpy as np

from .optimizer import Optimizer

class Freerex(Optimizer):
	"""docstring for Optimizer"""
	def __init__(self, k = 1):
		self.theta = None
		self.lt = None
		self.oneDivEtaSq = None
		self.at = None
		self.k = k
	
	def iterate(self, grad):
		gradnorm = np.linalg.norm(grad)

		if(self.theta is None):
			self.theta = -grad
		else:
			self.theta -= grad

		if(self.lt is None):
			self.lt = gradnorm
		else:
			self.lt = np.maximum(self.lt, gradnorm)

		gradSumNorm = np.linalg.norm(self.theta)

		if self.oneDivEtaSq is None:
			self.oneDivEtaSq = np.maximum(2 * gradnorm ** 2, self.lt * gradSumNorm)
		else:
			self.oneDivEtaSq = np.maximum(self.oneDivEtaSq + 2 * gradnorm ** 2, self.lt * gradSumNorm)

		if self.at is None:
			self.at = np.maximum(0.0, (1.0 / (self.lt ** 2)) * self.oneDivEtaSq)
		else:
			self.at = np.maximum(self.at, (1.0 / (self.lt ** 2)) * self.oneDivEtaSq)

		etat = np.sqrt(1.0 / self.oneDivEtaSq)
		return (self.theta / (self.at * gradSumNorm)) * (np.exp(etat * gradSumNorm / self.k) - 1)

	def clone(self):
		return Freerex(self.k)