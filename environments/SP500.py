"""
Class for tracking a ball in a video that passes through a box.
"""

import numpy as np
import numpy.random as random
from .environment import Environment

import os
from urllib.error import URLError
from urllib.request import urlretrieve
from pathlib import Path
import csv

class SP500(Environment):

	def __init__(self):
		self.initialized = False

	def initialize(self, params = {}):
		self.T = 0
		self.data = self.load_sp500()
		self.max_T = len(self.data) - 1
		self.initialized = True

	def load_sp500(self):
		values = []
		Path(os.path.join(".","datasets")).mkdir(parents=True, exist_ok=True)
		sp500csv = os.path.join(".","datasets","sp500.csv")
		sp500URL = "https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=520473600&period2=1545955200&interval=1d&events=history"
		if not os.path.exists(sp500csv):
			try:
				urlretrieve(sp500URL, sp500csv + ".tmp")
			except URLError:
				raise RuntimeError('Error downloading resource!')

			#delete columns
			with open(sp500csv + ".tmp") as csvfile:  
				inputFile = csv.reader(csvfile, delimiter=',')

				with open(sp500csv, 'w') as csvfileOut: 
					outputFile = csv.writer(csvfileOut, delimiter=',')

					for row in inputFile:
						outputFile.writerow([row[0],row[4]])
			try:
				os.remove(sp500csv + ".tmp")
			except:
				print("File was already deleted")

		with open(sp500csv) as csvfile:  
			plots = csv.reader(csvfile, delimiter=',')
			for row in list(plots)[1:]:
				values.append(float(row[1]) / 10000.0)
		return values

	def get_state_dim(self):
		return 1

	def get_action_dim(self):
		return 1

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

		return np.array([self.data[self.T - 1]])

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
				return np.array([self.data[self.T - 1]])

		if(1 <= t <= self.max_T):
			return np.array([self.data[t - 1]])

		return np.array([0])

	def getTimeSteps(self):
		return self.max_T

	def clone(self):
		return SP500()