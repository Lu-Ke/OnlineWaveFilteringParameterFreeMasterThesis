"""
Linear dynamical system base class
"""

import numpy as np
import tigercontrol
from tigercontrol.utils import generate_key
from .environment import Environment


class Setting4(Environment):
	"""
	Description: The base, master LDS class that all other LDS subenvironments inherit. 
		Simulates a linear dynamical system with a lot of flexibility and variety in
		terms of hyperparameter choices.
	"""

	def __init__(self):
		self.initialized = False

	def initialize(self, params):
	# action_dim=3, hidden_dim=2, out_dim = None, partially_observable = False, noise_distribution=None, \
	#	noise_magnitude=1.0, system_params = {}, initial_state = None
		"""
		Description: Randomly initialize the hidden dynamics of the system.
		Args:
			action_dim (int): Dimension of the actions.
			hidden_dim (int): Dimension of the hidden state.
			out_dim (int): Observation dimension. (Note: MUST specify partially_observable=True for d to be used!)
			partially_observable (bool): whether to project x to y
			noise_distribution (None, string, func): if None then no noise. Valid strings include ['normal', 'uniform']. 
				Valid noise functions must map inputs n (x dim), x (state), u (action), w (previous noise), and t (current time)
				to either a scalar or an n-dimensional vector of real values.
			noise magnitude (float): magnitude of noise
			params (dict): specify A, B, C, and D matrices in system dynamics
			initial_state (None, vector): initial x. If None then randomly initialized
		Returns:
			The first value in the time-series
		"""
		self.initialized = True
		self.T = 0
		self.action_dim = 1

		self.hidden_dim = 2

		self.out_dim = 1

		self.noise_magnitude = 1.0

		gaussian = lambda dims: np.random.normal(size=dims) * 0.05

		self.x = gaussian((self.hidden_dim,))

		#self.noise = lambda x, u: (0.0, 0.0)
		self.noise = lambda x, u: (gaussian((self.hidden_dim,)), gaussian((self.out_dim,)))


		self.A = np.array([[0.3, -0.1], [0.2, -0.25]])
		self.B = np.array([[-0.3], [0.4]])
		self.C = np.array([[0.5, 0.5]])
		self.D = np.array([[0.0]])



		def _step(x, u, eps):
			eps_x, eps_y = eps
			next_x = np.dot(self.A, x) + np.dot(self.B, u).flatten() + self.noise_magnitude * eps_x
			#y = np.dot(self.C, next_x) + np.dot(self.D, u) + self.noise_magnitude * eps_y
			y = np.dot(self.C, x) + np.dot(self.D, u) + self.noise_magnitude * eps_y
			return (next_x, y.flatten())
		self._step = _step

		u, w = np.zeros(self.action_dim), np.zeros(self.hidden_dim)
		# self.prev_noise = self.noise(n, self.x, u, w, self.T)
		self.prev_noise = self.noise(self.x, u)
		y = np.dot(self.C, self.x) + np.dot(self.D, u) + self.noise_magnitude * self.prev_noise[1] # (state_noise, obs_noise)
		return y

	def get_state_dim(self):
		return self.hidden_dim

	def get_action_dim(self):
		return self.action_dim

	def reset(self):
		return self.x

	def step(self, u):
		"""
		Description: Moves the system dynamics one time-step forward.
		Args:
			u (numpy.ndarray): control input, an n-dimensional real-valued vector.
		Returns:
			A new observation from the LDS.
		"""
		assert self.initialized
		self.T += 1
		if(self.T == 500):
			self.A = np.array([[0.3, 0.4], [0.1, -0.5]])
			self.B = np.array([[0.4], [-0.3]])
			self.C = np.array([[-0.3, 0.7]])
			self.D = np.array([[0.2]])

			#self.A = np.array([[-0.25, 0], [0.2, 0.15]])
			#self.B = np.array([[0.4], [-0.1]])
			#self.C = np.array([[0.7, 0.3]])
			#self.D = np.array([[0.4]])

		# self.prev_noise = self.noise(self.n, self.x, u, self.prev_noise, self.T) # change to x,u only
		self.prev_noise = self.noise(self.x, u)
		self.x, y = self._step(self.x, u, self.prev_noise)
		return y # even in fully observable case, y = self.x

	def hidden(self):
		"""
		Description: Return the hidden state of the system.
		Args:
			None
		Returns:
			h: The hidden state of the LDS.
		"""
		assert self.initialized
		return self.x

	def clone(self):
		return Setting4()