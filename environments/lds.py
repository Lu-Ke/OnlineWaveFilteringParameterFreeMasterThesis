"""
Linear dynamical system base class
"""

import numpy as np
import tigercontrol
from tigercontrol.utils import generate_key
from .environment import Environment


class LDS(Environment):
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
		if 'action_dim' in params:
			self.action_dim = params['action_dim']
		else:
			self.action_dim = 3


		if 'hidden_dim' in params:
			self.hidden_dim = params['hidden_dim']
		else:
			self.hidden_dim = 2

		if 'out_dim' in params:
			assert params['partially_observable']
			self.out_dim = params['out_dim']
		else:
			self.out_dim = None

		if 'noise_magnitude' in params:
			self.noise_magnitude = params['noise_magnitude']
		else:
			self.noise_magnitude = 1.0

		# random normal helper function
		#gaussian = lambda dims: np.random.normal(generate_key(), size=dims)

		gaussian = lambda dims: np.random.normal(size=dims)

		# initial state
		if 'initial_state' in params:
			self.x = params['initial_state']
		else:
			self.x = gaussian((self.hidden_dim,))

		system_params = params['system_params'].copy() # avoid overwriting system_params input dict

		# determine the noise function to use, allowing for conditioning on x, u, previous noise, and current t
		if not 'noise_distribution' in params:           # case no noise
			if params['partially_observable']:
				self.noise = lambda x, u: (0.0, 0.0)
			else:
				self.noise = lambda x, u: 0.0
		elif (params['noise_distribution'] == 'normal'):   # case normal distribution
			if params['partially_observable']:
				self.noise = lambda x, u: (gaussian((self.hidden_dim,)), gaussian((self.out_dim,)))
			else:
				self.noise = lambda x, u: gaussian((self.hidden_dim,))
		elif (params['noise_distribution'] == 'uniform'): # case uniform distribution
			if params['partially_observable']:
				self.noise = lambda x, u: (np.random.uniform(generate_key(), shape=(self.hidden_dim,), minval=-1, maxval=1), \
					random.uniform(generate_key(), shape=(self.out_dim,), minval=-1, maxval=1))
			else:
				self.noise = lambda x, u: np.random.uniform(generate_key(), shape=(self.hidden_dim,), minval=-1, maxval=1)
		else: # case custom function
			assert callable(params['noise_distribution']), "noise_distribution not valid input" # assert input is callable
			from inspect import getargspec
			arg_sub = getargspec(params['noise_distribution']).args # retrieve all parameters taken by provided function
			# for arg in arg_sub:
			#    assert arg in ['n', 'x', 'u', 'w', 't'], "noise_distribution takes invalid input"
			assert len(arg_sub) == 0 or len(arg_sub) == 2
			# noise_args = {'x': x, 'u': u}
			# arg_dict = {k:v for k,v in noise_args.items() if k in arg_sub}
			if len(arg_sub) == 0:
				self.noise = lambda x, u: noise_distribution()
			else:
				self.noise = lambda x, u: noise_distribution(x,u)

		# helper function that generates a random matrix with given dimensions
		for matrix, shape in {'A':(self.hidden_dim, self.hidden_dim), 'B':(self.hidden_dim, self.action_dim), \
							 'C':(self.out_dim, self.hidden_dim), 'D':(self.out_dim, self.action_dim)}.items():
			if matrix not in system_params: 
				if (params['out_dim'] == None) and (matrix == 'C' or matrix == 'D'): continue
				system_params[matrix] = gaussian(shape)
			else:
				assert system_params[matrix].shape == shape # check input has valid shape
		#normalize = lambda M, k: k * M / np.linalg.norm(M, ord=2) # scale largest eigenvalue to k
		#self.A = normalize(system_params['A'], 1.0)
		#self.B = normalize(system_params['B'], 1.0)
		#if params['partially_observable']:
		#	self.C = normalize(system_params['C'], 1.0)
		#	self.D = normalize(system_params['D'], 1.0)

		self.A = system_params['A']
		self.B = system_params['B']
		if params['partially_observable']:
			self.C = system_params['C']
			self.D = system_params['D']



		# different dynamics depending on whether the system is fully observable or not
		if params['partially_observable']:
			def _step(x, u, eps):
				eps_x, eps_y = eps
				next_x = np.dot(self.A, x) + np.dot(self.B, u).flatten() + self.noise_magnitude * eps_x
				#y = np.dot(self.C, next_x) + np.dot(self.D, u) + self.noise_magnitude * eps_y
				y = np.dot(self.C, x) + np.dot(self.D, u) + self.noise_magnitude * eps_y
				return (next_x, y.flatten())
			self._step = _step
		else:
			def _step(x, u, eps):
				eps_x = eps
				next_x = np.dot(self.A, x) + np.dot(self.B, u).flatten() + self.noise_magnitude * eps_x
				return (next_x, next_x)
			self._step = _step

		if params['partially_observable']: # return partially observable state
			u, w = np.zeros(self.action_dim), np.zeros(self.hidden_dim)
			# self.prev_noise = self.noise(n, self.x, u, w, self.T)
			self.prev_noise = self.noise(self.x, u)
			y = np.dot(self.C, self.x) + np.dot(self.D, u) + self.noise_magnitude * self.prev_noise[1] # (state_noise, obs_noise)
			return y
		self.prev_noise = np.zeros(n)

	def get_state_dim(self):
		return self.hidden_dim

	def get_action_dim(self):
		return self.action_dim

	def reset(self):
		return self.x # return true current state

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
		return LDS()