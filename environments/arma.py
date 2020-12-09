import numpy as np
import numpy.random as random

from .environment import Environment


class ARMA(Environment):
	"""
	Description: Simulates an autoregressive moving-average time-series.
	"""

	def __init__(self):
		self.initialized = False


	def initialize(self, params = {'T' : 10000}):
		self.T = 0
		self.max_T = params['T']
		self.n = 1
		self.a = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
		self.b = np.array([0.3, -0.2])
		self.timeSeries = random.randn(len(self.a))
		self.errorArray = np.empty(len(self.b))
		self.initError = random.randn(len(self.b)) * 0.3
		self.initialized = True
		self.newValue = self.timeSeries[-1]

	def get_state_dim(self):
		return 1

	def get_action_dim(self):
		return 1

	def reset(self):
		self.T = 0

	def step(self, u):
		"""
		Description: Moves the system dynamics one time-step forward.
		Args:
			u (numpy.ndarray): control input, an n-dimensional real-valued vector.
		Returns:
			A new observation from the ARMA model.
		"""
		assert self.initialized
		if(self.T < self.max_T):
			self.T += 1
		else:
			#maybe other behaviour is wanted here
			pass

		errorV = random.normal() * 0.3
		self.newValue = self.timeSeries.dot(self.a) + self.errorArray.dot(self.b) + errorV

		self.timeSeries = np.roll(self.timeSeries,-1)
		self.timeSeries[-1] = self.newValue
		
		self.errorArray = np.roll(self.errorArray,-1)
		self.errorArray[-1] = errorV
		
		return self.newValue

	def hidden(self):
		"""
		Description: Return the hidden state of the system.
		Args:
			None
		Returns:
			h: The hidden state of the LDS.
		"""
		assert self.initialized
		return (self.a, self.b)

	def getControlInput(self):
		assert self.initialized
		return self.newValue

	def getTimeSteps(self):
		return self.max_T

	def clone(self):
		return ARMA()
