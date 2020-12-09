# Predictor class 

class Predictor(object):

	def predict(self, action):
		""" Description: predicts output based on action """
		raise NotImplementedError

	def predictMultiple(self, actionList):
		"""
		Predict multiple steps at once without receiving true data in between.
		"""
		predicted = []
		for action in actionList:
			predicted.append(self.predict(action))
		return predicted

	def update_parameters(self, y_true):
		# update parameters according to given loss and update rule
		raise NotImplementedError

	def clone(self):
		print("You forgot the clone for the new predictor...")
		raise NotImplementedError

	def __str__(self):
		return '<{} instance>'.format(type(self).__name__)
		
	def __repr__(self):
		return self.__str__()