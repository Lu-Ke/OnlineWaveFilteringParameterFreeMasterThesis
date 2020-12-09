import numpy as np
from .predictor import Predictor
from .kalmanFilter import KalmanFilter
from utils.em import *

class EMKalmanFilter(Predictor):

	def __init__(self):
		self.initialized = False

	def initialize(self, params = {'data': 200, 'order': 2, 'iter': 500}):
		"""
		Description:
			Initialize the dynamics of the kalman filter.
		Args:
			h : estimate of h_0
			A : A
			B : B
			C : C
			P : initial estimate of error covariance P(0)
			Q : covariance of controller noise w(t)
			R : covariance of environment noise v(t)
		"""
		self.pred = KalmanFilter()
		self.dataCount = params['data']
		self.order = params['order']
		self.iter = params['iter']
		self.inputs = []
		self.outputs = []
		self.T = 0

	def update_parameters(self, y_true):
		if(self.T < self.dataCount):
			self.T += 1
			self.outputs.append(y_true)
			if(self.dataCount == self.T):
				#train
				A, B, C, D, P, Q, R, h0 = em(np.array(self.inputs).T, np.array(self.outputs).T, self.order, self.iter)
				print(A)
				print(B)
				print(C)
				print(D)
				print(P)
				print(Q)
				print(R)
				self.pred.initialize({'h': h0, 'A' : A, 'B': B, 'C': C, 'D': D, 'P': P, 'Q': Q, 'R': R})
			return -1
		else:
			return self.pred.update_parameters(y_true)

	def predict(self, action):
		if(self.T < self.dataCount):
			self.inputs.append(action)
			if(self.T == 0):
				return 0
			return self.outputs[-1]
		else:
			return self.pred.predict(action)



	def __str__(self):
		return "<EMKalmanFilter>"

	def clone(self):
		return EMKalmanFilter()