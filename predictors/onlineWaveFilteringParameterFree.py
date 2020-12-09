"""
 Parameter free version of the algorithm from paper 1.
"""

import numpy as np
from .predictor import Predictor
from .onlineWaveFilteringOnlyK import OnlineWaveFilteringOnlyK

class OnlineWaveFilteringParameterFree(Predictor):

	def __init__(self):
		self.initialized = False

	def initialize(self, params):
		self.T = params['timesteps']
		self.t = 0
		self.max_k = params['max_k']
		self.optimizer = params['opt'].clone()

		if('step_k' in params):
			self.step_k = params['step_k']
		else:
			self.step_k = 1
		self.predictors = []
		tempParams = {'timesteps': params['timesteps'], 'action_dim': params['action_dim'], 'out_dim': params['out_dim'], 'opt': params['optForSubPredictors'], 'k' : 1}
		for i in range(1, self.max_k + 1, self.step_k):
			tempParams['k'] = i
			tempPred = OnlineWaveFilteringOnlyK()
			tempPred.initialize(tempParams)
			self.predictors.append(tempPred)

		self.predictorCount = len(self.predictors)
		self.action_dim = params['action_dim']
		self.out_dim = params['out_dim']
		if('init' in params):
			if(params['init'] == 'equal'):
				self.weights = 1.0 / self.predictorCount * np.ones(self.predictorCount)
			else:
				self.weights = np.absolute(np.random.rand(self.predictorCount)) + 1e-6
		else:
			self.weights = np.absolute(np.random.rand(self.predictorCount)) + 1e-6
		self.weights /= np.sum(self.weights)
		self.y_pred = np.zeros(self.out_dim)

		self.initialized = True

	def predict(self, action):
		""" Description: returns action based on input state x """
		#store the new action

		predictions = np.zeros((self.out_dim, self.predictorCount))

		for i in range(self.predictorCount):
			predictions[:, i] = self.predictors[i].predict(action)

		self.t += 1

		self.y_pred = np.dot(predictions, self.weights)

		return self.y_pred



	def update_parameters(self, y_true):
		# update parameters according to given loss and update rule

		#update predictors
		differences = np.zeros(self.predictorCount)
		for i in range(self.predictorCount):
			differences[i] = self.predictors[i].update_parameters(y_true)

		y_diff = self.y_pred - y_true
		#print(str(y_true) + " vs " + str(self.y_pred))

		#print(differences, flush = True)
		#print(self.weights, flush = True)
		#print('---------------------------------------------', flush = True)

		loss = (y_diff ** 2).mean()

		#gradient update
		grad = 2 * differences

		self.weights = self.optimizer.iterate(grad)


		#print(str(y_true) + " vs " + str(self.y_pred))
		
		#if(np.absolute(np.sum(self.weights)) > 1e-6):
		"""
		if(loss > 100):
			print("Loss: " + str(loss))
			print(str(y_true) + " vs " + str(self.y_pred))
			print(differences)
			print(self.weights)
		"""
		#self.weights /= np.sum(self.weights)
		self.weights = np.absolute(self.weights) / np.sum(np.absolute(self.weights))
		#self.weights = np.absolute(self.weights)
		#if(np.sum(self.weights) > 1.0):
		#	self.weights = self.weights / np.sum(self.weights)
		"""
		if(loss > 100):
			print(self.weights)
		"""

		return loss

	def clone(self):
		return OnlineWaveFilteringParameterFree()