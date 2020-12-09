import numpy as np

class ActionGenerator(object):
	def __init__(self):
		super(ActionGenerator, self).__init__()

	def next(self, dim):
		pass

	def clone(self):
		print("You forgot the clone for the action ActionGenerator...")
		raise NotImplementedError

class RandomAction(ActionGenerator):
	def __init__(self, mu = 0, sigma = 1):
		super(RandomAction, self).__init__()
		self.sigma = sigma
		self.mu = mu

	def next(self, dim):
		return np.random.normal(size = dim) * self.sigma + self.mu

	def clone(self):
		return RandomAction()

class BlockAction(ActionGenerator):
	def __init__(self, prob_repeat = 0.5, mu = 0, sigma = 1):
		super(BlockAction, self).__init__()
		self.prob_repeat = prob_repeat
		self.last_block_action = None
		self.init = False
		self.mu = mu
		self.sigma = sigma

	def next(self, dim):
		if(self.init == False):
			self.init = True
			self.last_block_action = np.random.normal(size = dim) * self.sigma + self.mu
			return self.last_block_action

		if(np.random.uniform() > self.prob_repeat):
			#block changes
			self.last_block_action = np.random.normal(size = dim) * self.sigma + self.mu
		
		return self.last_block_action

	def clone(self):
		return BlockAction(self.prob_repeat)

class ProblemBasedAction(ActionGenerator):
	def __init__(self, problem):
		super(ProblemBasedAction, self).__init__()
		self.problem = problem.clone()
		self.problem.initialize()
		self.T = -1

	def next(self, dim):
		self.T += 1
		return self.problem.getControlInput(self.T)

	def clone(self):
		return ProblemBasedAction(self.problem)