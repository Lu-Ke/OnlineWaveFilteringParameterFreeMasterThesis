import numpy as np
from .predictor import Predictor

class ECVARMAOGD(Predictor):

	def __init__(self):
		self.initialized = False

	def initialize(self, params):
		self.p = params['p']
		self.action_dim = params['dim']
		self.lr = params['lr']

		self.rho = 1.0 / (np.log(params['timesteps']) ** 2)

		self.ts = np.zeros((self.p + 1, self.action_dim))

		self.Pi = np.random.rand(self.action_dim, self.action_dim)
		self.Gamma = np.random.rand(self.p, self.action_dim, self.action_dim)

		self.initialized = True

	def predict(self, action):
		""" Description: returns action based on input state x """
		#store the new action
		self.ts = np.roll(self.ts, 1, axis = 0)
		self.ts[0] = action

		last_term = np.zeros(self.action_dim)
		for i in range(self.p):
			last_term += self.Gamma[i].dot(self.ts[i] - self.ts[i + 1])
		self.y_pred = action + self.Pi.dot(action) + last_term


		return self.y_pred

	def update_parameters(self, y_true):
		y_diff = self.y_pred - y_true
		#print(str(y_true) + " vs " + str(self.y_pred))

		loss = (y_diff ** 2).mean()

		#update gamma
		for i in range(self.p):
			self.Gamma[i] = 2 * self.lr * np.dot(self.Gamma[i], y_diff)
			#project back
			self.Gamma[i] /= np.max(self.Gamma[i])

		self.Pi -= 2 * self.lr * np.dot(self.Pi, y_diff)

		self.Pi = self.nuclear_projection(self.Pi)

		return loss

	def simplex_projection(self, s):
		"""Projection onto the unit simplex."""
		if np.sum(s) <= self.rho and np.alltrue(s >= 0):
			return s
		# Code taken from https://gist.github.com/daien/1272551
		# get the array of cumulative sums of a sorted (decreasing) copy of v
		u = np.sort(s)[::-1]
		cssv = np.cumsum(u)
		# get the number of > 0 components of the optimal solution
		rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
		# compute the Lagrange multiplier associated to the simplex constraint
		theta = (cssv[rho] - 1) / (rho + 1.0)
		# compute the projection by thresholding v using theta
		return np.maximum(s-theta, 0)

	def nuclear_projection(self, A):
		"""Projection onto nuclear norm ball."""
		U, s, V = np.linalg.svd(A, full_matrices=False)
		s = self.simplex_projection(s)
		return U.dot(np.diag(s).dot(V))

	def clone(self):
		return ECVARMAOGD()