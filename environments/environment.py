import numpy as np

# class for online control tests
class Environment(object):
	def __init__(self):
		self.initialized = False

	def initialize(self, params):
		"""Initialize giving the dict of params"""
		raise NotImplementedError

	def reset(self):
		''' Description: reset environment to state at time 0, return state. '''
		raise NotImplementedError

	def step(self, **kwargs):
		''' Description: run one timestep of the environment's dynamics. '''
		raise NotImplementedError

	def get_out_dim(self):
		''' Description: return dimension of the state. '''
		raise NotImplementedError

	def get_action_dim(self):
		''' Description: return dimension of action inputs. '''
		raise NotImplementedError

	def close(self):
		''' Description: closes the environment and returns used memory '''
		pass

	def clone(self):
		print("You forgot the clone for the new environment...")
		raise NotImplementedError 

	def __str__(self):
		return '<{} instance>'.format(type(self).__name__)

	def __repr__(self):
		return self.__str__()