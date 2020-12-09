"""
https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

import numpy as np
import numpy.random as random
from .environment import Environment
import os
import csv

class RoomTemperature(Environment):

	def __init__(self):
		self.initialized = False

	def initialize(self, params = {}):
		self.T = 0
		self.data = self.load_roomTemp()
		self.max_T = 5000#len(self.data) - 1
		self.initialized = True

	def load_roomTemp(self):
		values = []
		roomtempcsv = os.path.join(".","datasets","room_temperature_humidity_all.csv")
		
		with open(roomtempcsv) as csvfile:  
			plots = csv.reader(csvfile, delimiter=';')
			for row in list(plots)[1:]:
				values.append(np.array([float(x) for x in row[1:]]) / 100.0)
		return values

	def get_state_dim(self):
		return self.data[0].shape[0]

	def get_action_dim(self):
		return self.data[0].shape[0]

	def reset(self):
		self.T = 0

	def step(self, u):
		"""
		Description: Moves the system dynamics one time-step forward.
		Args:
			u (numpy.ndarray): control input, an n-dimensional real-valued vector.
		Returns:
			A new observation from the LDS.
		"""
		assert self.initialized
		assert self.T < self.max_T
		self.T += 1

		return self.data[self.T - 1]

	def hidden(self):
		"""
		Description: Return the hidden state of the system.
		Args:
			None
		Returns:
			h: The hidden state of the LDS.
		"""
		assert self.initialized
		return self.data


	def getControlInput(self, t = -10):
		assert self.initialized
		if (t == -10):
			if(self.T < self.max_T):
				return self.data[self.T - 1]

		if(1 <= t <= self.max_T):
			return self.data[t - 1]

		return np.array([0] * self.get_action_dim())

	def getTimeSteps(self):
		return self.max_T

	def clone(self):
		return RoomTemperature()