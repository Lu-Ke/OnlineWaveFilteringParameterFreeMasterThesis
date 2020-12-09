import tigerforecast
import jax
import jax.numpy as np
import jax.experimental.stax as stax
from tigerforecast.utils.random import generate_key
from tigerforecast.methods import Method
from tigerforecast.utils.optimizers import * 
from tigerforecast.utils.optimizers.losses import mse
from .predictor import Predictor
from tigerforecast.methods import Method

#class ARIMA(Predictor):
class ARIMA(Method):
	"""
	Description: Implements the autoregressor extended to use the arima model.
	"""

	def __init__(self):
		self.initialized = False
		self.uses_regressors = False

	def initialize(self, params = {'n': 1, 'm': None, 'p': 3, 'd': 1, 'optimizer': OGD}):
		"""
		Description: Initializes autoregressive method parameters

		Args:
			n (int): dimension of the data
			p (int): Length of history used for prediction
			d (int): number of difference orders to use. For zero the original autoregressor is used.
			optimizer (class): optimizer choice
		"""
		self.initialized = True
		self.n = 1#params['n']
		self.p = params['p']
		self.d = params['d']

		self.past = np.zeros((self.p + self.d, self.n)) #store the last d x values to compute the differences

		glorot_init = stax.glorot() # returns a function that initializes weights

		# self.params = glorot_init(generate_key(), (p+1,1))
		self.params = {'phi' : glorot_init(generate_key(), (self.p + 1,1))}

		def _update_past(self_past, x):
			new_past = np.roll(self_past, 1)
			new_past = jax.ops.index_update(new_past, 0, x)
			return new_past
		self._update_past = jax.jit(_update_past)

		'''	unused (easy, but also inefficient version)
		def _computeDifference(x, d):
			if(d < 0):
				return 0.0
			if(d == 0):
				return x[0]
			else:
				return _computeDifference(x, d - 1) - _computeDifference(x[1:], d - 1)
		self._computeDifference = jax.jit(_computeDifference)'''

		def _getDifference(index, d, matrix):
			if(d < 0):
				return np.zeros(matrix[0, 0].shape)
			else:
				return matrix[d, index]
		self._getDifference = jax.jit(_getDifference)

		def _computeDifferenceMatrix(x, d):
			result = np.zeros((d + 1, x.shape[0], x.shape[1]))

			#first row (zeroth difference is the original time series)
			result = jax.ops.index_update(result, jax.ops.index[0, 0:len(x)], x)
			
			#fill the next rows
			for k in range(1, d + 1):
				result = jax.ops.index_update(result, jax.ops.index[k, 0 : len(x) - k - 1], result[k - 1, 0 : len(x) - k - 1] - result[k - 1, 1 : len(x) - k])

			return result
		self._computeDifferenceMatrix = jax.jit(_computeDifferenceMatrix)

		def _predict(params, x):
			phi = list(params.values())[0]
			differences = _computeDifferenceMatrix(x, self.d)
			x_plus_bias = np.vstack((np.ones((1, self.n)), np.array([_getDifference(i, self.d, differences) for i in range(0, self.p)])))
			return np.dot(x_plus_bias.T, phi).squeeze() + np.sum([_getDifference(0, k, differences) for k in range(self.d)])
		self._predict = jax.jit(_predict)

		def _getUpdateValues(x):
			diffMatrix = _computeDifferenceMatrix(x, self.d)
			differences = np.array([_getDifference(i, self.d, diffMatrix) for i in range(1, self.p + 1)])
			label = _getDifference(0, self.d, diffMatrix)
			return differences, label
		self._getUpdateValues = jax.jit(_getUpdateValues)


		self._store_optimizer(params['optimizer'], self._predict)


	def predict(self, action):
		"""
		Description: Predict next value given observation
		Args:
			x (int/numpy.ndarray): Observation
		Returns:
			Predicted value for the next time-step
		"""
		assert self.initialized, "ERROR: Method not initialized!"

		self.past = self._update_past(self.past, action) # squeeze to remove extra dimensions
		self.predicted = self._predict(self.params, self.past)
		return self.predicted

	def update_parameters(self, y_true):
		"""
		Description: Updates parameters using the specified optimizer
		Args:
			y (int/numpy.ndarray): True value at current time-step
		Returns:
			None
		"""
		assert self.initialized, "ERROR: Method not initialized!"

		differences, label = self._getUpdateValues(self.past)

		self.params = self.optimizer.update(self.params, differences, label)

		return ((self.predicted - y_true) ** 2).mean()

	def clone(self):
		return ARIMA()