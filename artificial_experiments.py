import numpy as np

from predictors.onlineWaveFiltering import OnlineWaveFiltering
from predictors.consistency import Consistency
from predictors.kalmanFilter import KalmanFilter
from predictors.emKalmanFilter import EMKalmanFilter
from predictors.ssidKalmanFilter import SSIDKalmanFilter
from predictors.onlineWaveFilteringParameterFree import OnlineWaveFilteringParameterFree
import matplotlib.pyplot as plt

#optimizers
from predictors.optimizers.ftrl import FTRL
from predictors.optimizers.hedge import Hedge

from environments.lds import LDS
from environments.setting4 import Setting4

from utils.plotter import Plotter
from utils.loss import quad_loss
from utils.experiment import Experiment
from utils.action_generation import *


def artif_exp_1():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(yscale = 'log', filt = "median, movingAverage", skip = 100, ylabel = "Average Squared Error", col = ['c', 'r', 'g', 'orange', 'k'])

	A = np.array([[0.2, 0], [0, -0.1]])
	B = np.array([[-0.3], [0.4]])
	C = np.array([[0.5, 0.5]])
	D = np.array([[0.0]])

	T = 1000

	ag = RandomAction(mu = 0, sigma = 0.3)

	exp.run_experiments_multiple(LDS(), {"timesteps" : T, 'action_dim': 1, 'hidden_dim': 2, 'out_dim': 1, 'partially_observable': True,\
								 'system_params': {'A': A, 'B' : B, 'C': C, 'D': D, 'noise_distribution': 'normal'}},\
						 [OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFiltering(),\
						  EMKalmanFilter(),\
						  SSIDKalmanFilter(),\
						  Consistency()],\
						 [{'timesteps': T, 'max_k' : 30, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'order': 2, 'data': 100, 'iter': 500},\
						  {'timesteps': T, 'order': 2, 'data': 100},\
						  {'timesteps': T, 'out_dim': 1}],\
						 ['OnlineWaveFilteringParameterFree',\
						  'OnlineWaveFiltering',\
						  'EM',\
						  '4SID',\
						  'Consistency'],\
						 n_runs = 20,\
						 plotter = plotting,\
						 action_generator = ag,\
						 verbose = True)


def artif_exp_2():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(yscale = 'log', filt = "median, movingAverage", skip = 100, ylabel = "Average Squared Error", col = ['c', 'r', 'g', 'orange', 'k'])

	A = np.array([[0.999, 0], [0, 0.5]])
	B = np.array([[1.0], [1.0]])
	C = np.array([[1.0, 1.0]])
	D = np.array([[0.0]])

	T = 1000

	ag = RandomAction(mu = 0, sigma = 0.3)

	exp.run_experiments_multiple(LDS(), {"timesteps" : T, 'action_dim': 1, 'hidden_dim': 2, 'out_dim': 1, 'partially_observable': True,\
								 'system_params': {'A': A, 'B' : B, 'C': C, 'D': D, 'noise_distribution': 'normal'}},\
						 [OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFiltering(),\
						  EMKalmanFilter(),\
						  SSIDKalmanFilter(),\
						  Consistency()],\
						 [{'timesteps': T, 'max_k' : 30, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'order': 2, 'data': 100, 'iter': 500},\
						  {'timesteps': T, 'order': 2, 'data': 100},\
						  {'timesteps': T, 'out_dim': 1}],\
						 ['OnlineWaveFilteringParameterFree',\
						  'OnlineWaveFiltering',\
						  'EM',\
						  '4SID',\
						  'Consistency'],\
						 n_runs = 20,\
						 plotter = plotting,\
						 action_generator = ag,\
						 verbose = True)

def artif_exp_3():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(yscale = 'log', filt = "median, movingAverage", skip = 100, ylabel = "Average Squared Error", col = ['c', 'r', 'g', 'orange', 'k'])

	A = np.array([[0.3, -0.1], [0.2, -0.25]])
	B = np.array([[-0.3], [0.4]])
	C = np.array([[0.5, 0.5]])
	D = np.array([[0.0]])

	T = 1000

	ag = RandomAction(mu = 0, sigma = 0.3)

	exp.run_experiments_multiple(LDS(), {"timesteps" : T, 'action_dim': 1, 'hidden_dim': 2, 'out_dim': 1, 'partially_observable': True,\
								 'system_params': {'A': A, 'B' : B, 'C': C, 'D': D, 'noise_distribution': 'normal'}},\
						 [OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFiltering(),\
						  EMKalmanFilter(),\
						  SSIDKalmanFilter(),\
						  Consistency()],\
						 [{'timesteps': T, 'max_k' : 30, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'order': 2, 'data': 100, 'iter': 500},\
						  {'timesteps': T, 'order': 2, 'data': 100},\
						 {'timesteps': T, 'out_dim': 1}],\
						 ['OnlineWaveFilteringParameterFree',\
						  'OnlineWaveFiltering',\
						  'EM',\
						  '4SID',\
						  'Consistency'],\
						 n_runs = 20,\
						 plotter = plotting,\
						 action_generator = ag,\
						 verbose = True)

def artif_exp_4():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(yscale = 'log', filt = "median, movingAverage", skip = 100, ylabel = "Average Squared Error", col = ['c', 'r', 'g', 'orange', 'k'])

	T = 1000

	ag = RandomAction(mu = 0, sigma = 0.3)

	exp.run_experiments_multiple(Setting4(), {"timesteps" : T},\
						 [OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFiltering(),\
						  EMKalmanFilter(),\
						  SSIDKalmanFilter(),\
						  Consistency()],\
						 [{'timesteps': T, 'max_k' : 30, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'order': 2, 'data': 100, 'iter': 500},\
						  {'timesteps': T, 'order': 2, 'data': 100},\
						  {'timesteps': T, 'out_dim': 1}],\
						 ['OnlineWaveFilteringParameterFree',\
						  'OnlineWaveFiltering',\
						  'EM',\
						  '4SID',\
						  'Consistency'],\
						 n_runs = 20,\
						 plotter = plotting,\
						 action_generator = ag,\
						 verbose = True)

def artif_exp_5():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(yscale = 'log', filt = "median, movingAverage", skip = 100, ylabel = "Average Squared Error", col = ['c', 'r', 'g', 'k'])

	A = np.diag([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	B = np.eye(10)
	C = np.random.normal(size = (10, 10)) * 0.3
	D = np.zeros((10, 10))

	T = 1000

	ag = BlockAction(prob_repeat = 0.8, sigma = 0.3)

	exp.run_experiments_multiple(LDS(), {'action_dim': 10, 'hidden_dim': 10, 'out_dim': 10, 'partially_observable': True,\
								 'system_params': {'A': A, 'B' : B, 'C': C, 'D': D}},\
						 [OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFiltering(),\
						  EMKalmanFilter(),\
						  Consistency()],\
						 [{'timesteps': T, 'max_k' : 30, 'action_dim': 10, 'out_dim': 10, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 5},\
						  {'timesteps': T, 'order': 10, 'data': 100, 'iter': 500},\
						  {'timesteps': T, 'out_dim': 10}],\
						 ['OnlineWaveFilteringParameterFree',\
						  'OnlineWaveFiltering',\
						  'EM',\
						  'Consistency'],\
						 action_generator = ag,\
						 n_runs = 20,\
						 plotter = plotting,\
						 verbose = True)

if __name__ == '__main__':
	artif_exp_1()
	artif_exp_2()
	artif_exp_3()
	artif_exp_4()
	artif_exp_5()