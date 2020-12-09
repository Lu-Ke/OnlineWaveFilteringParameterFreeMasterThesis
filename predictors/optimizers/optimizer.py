
class Optimizer(object):
	"""docstring for Optimizer"""
	def __init__(self):
		pass
	
	def iterate(self, grad):
		raise NotImplementedError

	def clone(self):
		print("You forgot the clone for the new optimizer...")
		raise NotImplementedError