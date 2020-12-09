"""
 Parameter free version of the algorithm from paper 1.
"""

import numpy as np
import scipy as sp
from scipy import linalg as spl
from .predictor import Predictor


class OnlineWaveFilteringOnlyK(Predictor):

	def __init__(self):
		self.initialized = False

	def initialize(self, params):
		self.T = params['timesteps']
		self.t = 0
		self.k = params['k']
		self.optimizer = params['opt'].clone()
		self.eigenvalues, self.eigenvectors = self._compute_eigenpairs()
		self.action_dim = params['action_dim']
		self.out_dim = params['out_dim']
		self.k_prime = self.action_dim * (self.k + 2) + self.out_dim
		self.M = np.random.rand(self.out_dim, self.k_prime)
		self.y_pred = np.zeros(self.out_dim)
		self.all_actions = np.zeros((self.T + self.k, self.action_dim))
		self.last_output = np.zeros(self.out_dim)

		self.initialized = True

	def predict(self, action):
		""" Description: returns action based on input state x """
		#store the new action
		self.all_actions[self.t + self.k, :] = action

		Vx = np.dot(self.eigenvectors.T, self.all_actions[self.t : self.t + self.k])
		Xtilde_ij = np.einsum('ij,i->ij', Vx, self.eigenvalues).reshape(-1,1)
		self.X_approx = np.concatenate((Xtilde_ij, 
								self.all_actions[self.t - 1 + self.k : self.t - 1 + self.k + 2].flatten().reshape(-1, 1),
								self.last_output.reshape(-1, 1)), axis=0)

		self.t += 1

		self.y_pred = np.dot(self.M, self.X_approx).flatten()

		return self.y_pred

	def update_parameters(self, y_true):
		# update parameters according to given loss and update rule

		self.last_output = y_true

		#starting from line 7 replace the OPGD with FTRL
		#y_diff = y_true - self.y_pred
		y_diff = self.y_pred - y_true
		#print(str(y_true) + " vs " + str(self.y_pred))

		loss = (y_diff ** 2).mean()

		#gradient update
		grad = 2 * np.outer(y_diff.reshape(-1, 1), self.X_approx)

		self.M = self.optimizer.iterate(grad)

		return np.mean(y_diff)

	def _compute_eigenpairs(self):
		"""Calculate the eigenvectors and eigenvalues of the Hankel matrix"""
		z = np.array([ [2.0 / (np.power(i + j, 3) - (i + j)) for j in range(1, self.k + 1)] for i in range(1, self.k + 1)])
		z += 1e-8 * np.eye(self.k)
		
		eigVals, eigVecs = spl.eigh(z, check_finite = False)

		#idx = eigVals.argsort()[::-1]
		#eigVals = eigVals[idx]
		#eigVecs = eigVecs[:,idx]

		return np.power(eigVals, 0.25), eigVecs

	def clone(self):
		return OnlineWaveFilteringOnlyK()