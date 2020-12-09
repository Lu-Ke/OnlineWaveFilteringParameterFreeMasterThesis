import matplotlib.pyplot as plt
import numpy as np

import scipy.signal as sp

class Plotter(object):
	"""Class for plotting data from an experiment"""
	def __init__(self):
		self.initialized = False
	
	def initialize(self, xlabel = "step", ylabel = "loss", save = "", show = True, xscale = 'linear', yscale = 'linear', filt = "", skip = 0, printLast = False, col = None):
		self.initialized = True
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.save = save
		self.show = show
		self.xscale = xscale
		self.yscale = yscale
		self.filter = filt
		self.skip = skip
		self.printLast = printLast
		self.col = col

	def plot(self, loss_list, names = None):
		for i in range(len(loss_list)):
			name = i
			if(names != None and len(names) > i):
				name = names[i]

			if('median' in self.filter):
				loss_list[i] = sp.medfilt(loss_list[i], 5)
			if('movingAverage' in self.filter):
				curr_avg = 0
				for j in range(len(loss_list)):
					curr_avg = (j / (j + 1)) * curr_avg + loss_list[j] / (j + 1)
					loss_list[j] = curr_avg

			if(not self.col is None):
				plt.plot(range(self.skip, len(loss_list[i])), loss_list[i][self.skip:], label = name, c = self.col[i])
			else:
				plt.plot(range(self.skip, len(loss_list[i])), loss_list[i][self.skip:], label = name)

		plt.xlabel(self.xlabel)
		plt.ylabel(self.ylabel)
		plt.xscale(self.xscale)
		plt.yscale(self.yscale)

		if(names != None):
			plt.legend()
			if(self.printLast):
				for i in range(len(names)):
					print(names[i] + ": " + str(loss_list[i][-1]))
				print("------------------------------------")
		if(self.save != ""):
			plt.savefig(self.save)
		if(self.show):
			plt.show()