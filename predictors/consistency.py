"""
 Algorithm from paper 1.
"""

import numpy as np
import scipy as sp
from .predictor import Predictor

class Consistency(Predictor):

	def __init__(self):
		self.initialized = False

	def initialize(self, params):
		self.out_dim = params['out_dim']
		self.last_y = np.zeros(self.out_dim)
		self.initialized = True

	def predict(self, action):
		""" Description: returns action based on input state x """

		return self.last_y

	def update_parameters(self, y_true):
		# update parameters according to given loss and update rule
		loss = np.linalg.norm(y_true - self.last_y) ** 2
		self.last_y = y_true
		return loss

	def clone(self):
		return Consistency()