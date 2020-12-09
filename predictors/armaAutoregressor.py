import tigerforecast
import jax
import jax.numpy as np
import jax.experimental.stax as stax
from tigerforecast.utils.random import generate_key
from tigerforecast.utils.optimizers import * 
from tigerforecast.utils.optimizers.losses import mse
from .predictor import Predictor
from tigerforecast.methods import Method

#class ARMA(Predictor):
class ARMA(Method):
	"""
	Description: Implements the equivalent of an AR(p) method - predicts a linear
	combination of the previous p observed values in a time-series
	"""

	def __init__(self):
		self.initialized = False
		self.uses_regressors = False

	def initialize(self, params = {'n': 1, "m": None, "p": 3, "optimizer": OGD}):
		"""
		Description: Initializes autoregressive method parameters

		Args:
			p (int): Length of history used for prediction
			optimizer (class): optimizer choice
			loss (class): loss choice
			lr (float): learning rate for update
		"""
		self.initialized = True
		self.n = 1#params['n']
		self.p = params['p']

		self.past = np.zeros((self.p, self.n))

		glorot_init = stax.glorot() # returns a function that initializes weights

		# self.params = glorot_init(generate_key(), (p+1,1))
		self.params = {'phi' : glorot_init(generate_key(), (self.p, 1))}

		def _update_past(self_past, x):
			new_past = np.roll(self_past, self.n)
			new_past = jax.ops.index_update(new_past, 0, x)
			return new_past
		self._update_past = jax.jit(_update_past)

		def _predict(params, x):
			phi = list(params.values())[0]
			return np.dot(x.T, phi).squeeze()
		self._predict = jax.jit(_predict)

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

		self.params = self.optimizer.update(self.params, self.past, y_true)

		return ((self.predicted - y_true) ** 2).mean()

	def clone(self):
		return ARMA()