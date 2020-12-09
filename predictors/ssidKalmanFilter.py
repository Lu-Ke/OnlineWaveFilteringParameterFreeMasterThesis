import numpy as np
from .predictor import Predictor
from .kalmanFilter import KalmanFilter
from utils.det4sid import *

class SSIDKalmanFilter(Predictor):

	def __init__(self):
		self.initialized = False

	def initialize(self, params = {'data': 200, 'order': 2}):
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
		self.inputs = []
		self.outputs = []
		self.T = 0

	def update_parameters(self, y_true):
		if(self.T < self.dataCount):
			self.T += 1
			self.outputs.append(y_true)
			if(self.dataCount == self.T):
				#train
				A, B, C, D = det4sid(np.array(self.inputs).T, np.array(self.outputs).T, self.order)
				#print(A)
				#print(B)
				#print(C)
				#print(D)
				h0 = np.array([0] * self.order)
				P = np.eye(self.order) * 0.3
				Q = np.eye(self.order) * 0.3
				R = np.eye(y_true.shape[0]) * 0.3
				self.pred.initialize({'h': h0, 'A' : A, 'B': B, 'C': C, 'D': D, 'P': P, 'Q': Q, 'R': R})
			return -1
		else:
			#print(y_true)
			return self.pred.update_parameters(y_true)

	def predict(self, action):
		if(self.T < self.dataCount):
			self.inputs.append(action)
			if(self.T == 0):
				return 0
			return self.outputs[-1]
		else:
			predic = self.pred.predict(action)
			#print(predic)
			return predic#self.pred.predict(action)



	def __str__(self):
		return "<SSIDKalmanFilter>"

	def clone(self):
		return SSIDKalmanFilter()