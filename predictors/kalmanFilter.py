import numpy as np
from .predictor import Predictor

class KalmanFilter(Predictor):
	"""
	h = A * h + B * action
	pred = C * h
	"""

	def __init__(self):
		self.initialized = False

	def initialize(self, params):
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
		self.h, self.A, self.B, self.C, self.D  = params['h'], params['A'], params['B'], params['C'], params['D']
		self.P, self.Q, self.R = params['P'], params['Q'], params['R']

		self.last_y_true = np.zeros(self.C.shape[0])

		self.K = np.zeros(self.C.shape)

		#print(self.K.shape)
		#exit()
		
	def update_parameters(self, y_true):

		self.h = self.h + np.dot(self.K, (y_true - np.dot(self.C, self.h)))
		self.P = self.P - np.dot(self.K, np.dot(self.C, self.P))

		self.last_y_true = y_true

		return ((self.last_pred - y_true) ** 2).mean()

	def predict(self, action):

		# compute prediction for y
		self.K = np.dot(np.dot(self.P, self.C.T), np.linalg.inv(np.dot(np.dot(self.C, self.P), self.C.T) + self.R))
		self.last_pred = self.h + np.dot(self.K, (self.last_y_true - np.dot(self.C, self.h))) + self.D.dot(action)

		# update hidden state
		self.h = np.dot(self.A, self.h) + np.dot(self.B, action)
		self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

		return self.last_pred


	def __str__(self):
		return "<KalmanFilter>"

	def clone(self):
		return KalmanFilter()