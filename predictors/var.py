import numpy as np
from .predictor import Predictor
from statsmodels.tsa.vector_ar.var_model import VAR as VARIntern
from random import random

class VAR(Predictor):

	def __init__(self):
		self.initialized = False

	def initialize(self, params):
		self.p = params['p']
		self.action_dim = params['dim']

		self.ts = [np.zeros(self.action_dim) for i in range(self.p)]#np.zeros((self.p, self.action_dim))

		self.initialized = True

	def predict(self, action):
		""" Description: returns action based on input state x """
		#store the new action
		#self.ts = np.roll(self.ts, -1, axis = 0)
		#self.ts[-1] = action
		del self.ts[0]
		self.ts.append(action)

		model = VARIntern(self.ts)
		model_fit = model.fit(trend = 'n')
		self.y_pred = model_fit.forecast(model_fit.endog, steps=1)

		return self.y_pred

	def update_parameters(self, y_true):

		return ((self.y_pred - y_true) ** 2).mean()

	def clone(self):
		return VAR()