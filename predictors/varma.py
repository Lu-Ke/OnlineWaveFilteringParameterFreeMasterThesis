import numpy as np
from .predictor import Predictor
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random

class VARMA(Predictor):

	def __init__(self):
		self.initialized = False

	def initialize(self, params):
		self.p = params['p']
		self.action_dim = params['dim']

		self.ts = [[0] * self.action_dim] * self.p#[np.zeros(self.action_dim) for i in range(self.p)]#np.zeros((self.p, self.action_dim))


		data = list()
		for i in range(100):
			v1 = random()
			v2 = v1 + random()
			row = [v1, v2]
			data.append(row)
		model = VARMAX(self.ts, order=(16, 16))
		print("VARMAX")
		model_fit = model.fit()
		print("fit")
		exit()

		self.initialized = True

	def predict(self, action):
		""" Description: returns action based on input state x """
		#store the new action
		#self.ts = np.roll(self.ts, -1, axis = 0)
		#self.ts[-1] = action
		del self.ts[0]
		self.ts.append(action)
		#print(self.ts)

		model = VARMAX(self.ts, order=(self.p, self.p))
		model_fit = model.fit(disp=False)
		self.y_pred = model_fit.forecast(steps = 1)

		print(self.y_pred)

		return self.y_pred

	def update_parameters(self, y_true):

		return ((self.y_pred - y_true) ** 2).mean()

	def clone(self):
		return VARMA()