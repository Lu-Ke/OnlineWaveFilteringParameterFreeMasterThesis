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

def tune_exp_1():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(yscale = 'log', filt = "median, movingAverage", skip = 0, printLast = True, show = True)

	A = np.array([[0.2, 0], [0, -0.1]])
	B = np.array([[-0.3], [0.4]])
	C = np.array([[0.5, 0.5]])
	D = np.array([[0.0]])

	T = 100

	ag = RandomAction(mu = 0, sigma = 0.3)

	exp.run_experiments_multiple(LDS(), {"timesteps" : T, 'action_dim': 1, 'hidden_dim': 2, 'out_dim': 1, 'partially_observable': True,\
								 'system_params': {'A': A, 'B' : B, 'C': C, 'D': D, 'noise_distribution': 'normal'}},\
						 [OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering()
						 ],\
						 [{'timesteps': T, 'max_k' : 50, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0}],\
						 ['Reference',\
						 'OnlineWaveFiltering_10_4_1',\
						 'OnlineWaveFiltering_10_3_1',\
						 'OnlineWaveFiltering_10_2_1',\
						 'OnlineWaveFiltering_10_1_1',\
						 'OnlineWaveFiltering_10_0_1',\
						 'OnlineWaveFiltering_30_4_1',\
						 'OnlineWaveFiltering_30_3_1',\
						 'OnlineWaveFiltering_30_2_1',\
						 'OnlineWaveFiltering_30_1_1',\
						 'OnlineWaveFiltering_30_0_1',\
						 'OnlineWaveFiltering_50_4_1',\
						 'OnlineWaveFiltering_50_3_1',\
						 'OnlineWaveFiltering_50_2_1',\
						 'OnlineWaveFiltering_50_1_1',\
						 'OnlineWaveFiltering_50_0_1',\
						 'OnlineWaveFiltering_10_4_3',\
						 'OnlineWaveFiltering_10_3_3',\
						 'OnlineWaveFiltering_10_2_3',\
						 'OnlineWaveFiltering_10_1_3',\
						 'OnlineWaveFiltering_10_0_3',\
						 'OnlineWaveFiltering_30_4_3',\
						 'OnlineWaveFiltering_30_3_3',\
						 'OnlineWaveFiltering_30_2_3',\
						 'OnlineWaveFiltering_30_1_3',\
						 'OnlineWaveFiltering_30_0_3',\
						 'OnlineWaveFiltering_50_4_3',\
						 'OnlineWaveFiltering_50_3_3',\
						 'OnlineWaveFiltering_50_2_3',\
						 'OnlineWaveFiltering_50_1_3',\
						 'OnlineWaveFiltering_50_0_3',\
						 'OnlineWaveFiltering_10_4_5',\
						 'OnlineWaveFiltering_10_3_5',\
						 'OnlineWaveFiltering_10_2_5',\
						 'OnlineWaveFiltering_10_1_5',\
						 'OnlineWaveFiltering_10_0_5',\
						 'OnlineWaveFiltering_30_4_5',\
						 'OnlineWaveFiltering_30_3_5',\
						 'OnlineWaveFiltering_30_2_5',\
						 'OnlineWaveFiltering_30_1_5',\
						 'OnlineWaveFiltering_30_0_5',\
						 'OnlineWaveFiltering_50_4_5',\
						 'OnlineWaveFiltering_50_3_5',\
						 'OnlineWaveFiltering_50_2_5',\
						 'OnlineWaveFiltering_50_1_5',\
						 'OnlineWaveFiltering_50_0_5',\
						 'OnlineWaveFiltering_10_4_7',\
						 'OnlineWaveFiltering_10_3_7',\
						 'OnlineWaveFiltering_10_2_7',\
						 'OnlineWaveFiltering_10_1_7',\
						 'OnlineWaveFiltering_10_0_7',\
						 'OnlineWaveFiltering_30_4_7',\
						 'OnlineWaveFiltering_30_3_7',\
						 'OnlineWaveFiltering_30_2_7',\
						 'OnlineWaveFiltering_30_1_7',\
						 'OnlineWaveFiltering_30_0_7',\
						 'OnlineWaveFiltering_50_4_7',\
						 'OnlineWaveFiltering_50_3_7',\
						 'OnlineWaveFiltering_50_2_7',\
						 'OnlineWaveFiltering_50_1_7',\
						 'OnlineWaveFiltering_50_0_7'],
						 n_runs = 5,\
						 plotter = plotting,\
						 action_generator = ag,\
						 verbose = False)


def tune_exp_2():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(yscale = 'log', filt = "median, movingAverage", skip = 0, printLast = True, show = False)

	A = np.array([[0.999, 0], [0, 0.5]])
	B = np.array([[1.0], [1.0]])
	C = np.array([[1.0, 1.0]])
	D = np.array([[0.0]])

	T = 100

	ag = RandomAction(mu = 0, sigma = 0.3)

	exp.run_experiments_multiple(LDS(), {"timesteps" : T, 'action_dim': 1, 'hidden_dim': 2, 'out_dim': 1, 'partially_observable': True,\
								 'system_params': {'A': A, 'B' : B, 'C': C, 'D': D}},\
						 [OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering()
						 ],\
						 [{'timesteps': T, 'max_k' : 50, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0}],\
						 ['Reference',\
						 'OnlineWaveFiltering_10_4_1',\
						 'OnlineWaveFiltering_10_3_1',\
						 'OnlineWaveFiltering_10_2_1',\
						 'OnlineWaveFiltering_10_1_1',\
						 'OnlineWaveFiltering_10_0_1',\
						 'OnlineWaveFiltering_30_4_1',\
						 'OnlineWaveFiltering_30_3_1',\
						 'OnlineWaveFiltering_30_2_1',\
						 'OnlineWaveFiltering_30_1_1',\
						 'OnlineWaveFiltering_30_0_1',\
						 'OnlineWaveFiltering_50_4_1',\
						 'OnlineWaveFiltering_50_3_1',\
						 'OnlineWaveFiltering_50_2_1',\
						 'OnlineWaveFiltering_50_1_1',\
						 'OnlineWaveFiltering_50_0_1',\
						 'OnlineWaveFiltering_10_4_3',\
						 'OnlineWaveFiltering_10_3_3',\
						 'OnlineWaveFiltering_10_2_3',\
						 'OnlineWaveFiltering_10_1_3',\
						 'OnlineWaveFiltering_10_0_3',\
						 'OnlineWaveFiltering_30_4_3',\
						 'OnlineWaveFiltering_30_3_3',\
						 'OnlineWaveFiltering_30_2_3',\
						 'OnlineWaveFiltering_30_1_3',\
						 'OnlineWaveFiltering_30_0_3',\
						 'OnlineWaveFiltering_50_4_3',\
						 'OnlineWaveFiltering_50_3_3',\
						 'OnlineWaveFiltering_50_2_3',\
						 'OnlineWaveFiltering_50_1_3',\
						 'OnlineWaveFiltering_50_0_3',\
						 'OnlineWaveFiltering_10_4_5',\
						 'OnlineWaveFiltering_10_3_5',\
						 'OnlineWaveFiltering_10_2_5',\
						 'OnlineWaveFiltering_10_1_5',\
						 'OnlineWaveFiltering_10_0_5',\
						 'OnlineWaveFiltering_30_4_5',\
						 'OnlineWaveFiltering_30_3_5',\
						 'OnlineWaveFiltering_30_2_5',\
						 'OnlineWaveFiltering_30_1_5',\
						 'OnlineWaveFiltering_30_0_5',\
						 'OnlineWaveFiltering_50_4_5',\
						 'OnlineWaveFiltering_50_3_5',\
						 'OnlineWaveFiltering_50_2_5',\
						 'OnlineWaveFiltering_50_1_5',\
						 'OnlineWaveFiltering_50_0_5',\
						 'OnlineWaveFiltering_10_4_7',\
						 'OnlineWaveFiltering_10_3_7',\
						 'OnlineWaveFiltering_10_2_7',\
						 'OnlineWaveFiltering_10_1_7',\
						 'OnlineWaveFiltering_10_0_7',\
						 'OnlineWaveFiltering_30_4_7',\
						 'OnlineWaveFiltering_30_3_7',\
						 'OnlineWaveFiltering_30_2_7',\
						 'OnlineWaveFiltering_30_1_7',\
						 'OnlineWaveFiltering_30_0_7',\
						 'OnlineWaveFiltering_50_4_7',\
						 'OnlineWaveFiltering_50_3_7',\
						 'OnlineWaveFiltering_50_2_7',\
						 'OnlineWaveFiltering_50_1_7',\
						 'OnlineWaveFiltering_50_0_7'],
						 n_runs = 5,\
						 plotter = plotting,\
						 action_generator = ag,\
						 verbose = False)

def tune_exp_3():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(yscale = 'log', filt = "median, movingAverage", skip = 0, printLast = True, show = False)

	A = np.array([[0.3, -0.1], [0.2, -0.25]])
	B = np.array([[-0.3], [0.4]])
	C = np.array([[0.5, 0.5]])
	D = np.array([[0.0]])

	T = 100

	ag = RandomAction(mu = 0, sigma = 0.3)

	exp.run_experiments_multiple(LDS(), {"timesteps" : T, 'action_dim': 1, 'hidden_dim': 2, 'out_dim': 1, 'partially_observable': True,\
								 'system_params': {'A': A, 'B' : B, 'C': C, 'D': D}},\
						 [OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering()
						 ],\
						 [{'timesteps': T, 'max_k' : 50, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0}],\
						 ['Reference',\
						 'OnlineWaveFiltering_10_4_1',\
						 'OnlineWaveFiltering_10_3_1',\
						 'OnlineWaveFiltering_10_2_1',\
						 'OnlineWaveFiltering_10_1_1',\
						 'OnlineWaveFiltering_10_0_1',\
						 'OnlineWaveFiltering_30_4_1',\
						 'OnlineWaveFiltering_30_3_1',\
						 'OnlineWaveFiltering_30_2_1',\
						 'OnlineWaveFiltering_30_1_1',\
						 'OnlineWaveFiltering_30_0_1',\
						 'OnlineWaveFiltering_50_4_1',\
						 'OnlineWaveFiltering_50_3_1',\
						 'OnlineWaveFiltering_50_2_1',\
						 'OnlineWaveFiltering_50_1_1',\
						 'OnlineWaveFiltering_50_0_1',\
						 'OnlineWaveFiltering_10_4_3',\
						 'OnlineWaveFiltering_10_3_3',\
						 'OnlineWaveFiltering_10_2_3',\
						 'OnlineWaveFiltering_10_1_3',\
						 'OnlineWaveFiltering_10_0_3',\
						 'OnlineWaveFiltering_30_4_3',\
						 'OnlineWaveFiltering_30_3_3',\
						 'OnlineWaveFiltering_30_2_3',\
						 'OnlineWaveFiltering_30_1_3',\
						 'OnlineWaveFiltering_30_0_3',\
						 'OnlineWaveFiltering_50_4_3',\
						 'OnlineWaveFiltering_50_3_3',\
						 'OnlineWaveFiltering_50_2_3',\
						 'OnlineWaveFiltering_50_1_3',\
						 'OnlineWaveFiltering_50_0_3',\
						 'OnlineWaveFiltering_10_4_5',\
						 'OnlineWaveFiltering_10_3_5',\
						 'OnlineWaveFiltering_10_2_5',\
						 'OnlineWaveFiltering_10_1_5',\
						 'OnlineWaveFiltering_10_0_5',\
						 'OnlineWaveFiltering_30_4_5',\
						 'OnlineWaveFiltering_30_3_5',\
						 'OnlineWaveFiltering_30_2_5',\
						 'OnlineWaveFiltering_30_1_5',\
						 'OnlineWaveFiltering_30_0_5',\
						 'OnlineWaveFiltering_50_4_5',\
						 'OnlineWaveFiltering_50_3_5',\
						 'OnlineWaveFiltering_50_2_5',\
						 'OnlineWaveFiltering_50_1_5',\
						 'OnlineWaveFiltering_50_0_5',\
						 'OnlineWaveFiltering_10_4_7',\
						 'OnlineWaveFiltering_10_3_7',\
						 'OnlineWaveFiltering_10_2_7',\
						 'OnlineWaveFiltering_10_1_7',\
						 'OnlineWaveFiltering_10_0_7',\
						 'OnlineWaveFiltering_30_4_7',\
						 'OnlineWaveFiltering_30_3_7',\
						 'OnlineWaveFiltering_30_2_7',\
						 'OnlineWaveFiltering_30_1_7',\
						 'OnlineWaveFiltering_30_0_7',\
						 'OnlineWaveFiltering_50_4_7',\
						 'OnlineWaveFiltering_50_3_7',\
						 'OnlineWaveFiltering_50_2_7',\
						 'OnlineWaveFiltering_50_1_7',\
						 'OnlineWaveFiltering_50_0_7'],
						 n_runs = 5,\
						 plotter = plotting,\
						 action_generator = ag,\
						 verbose = False)

def tune_exp_4():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(yscale = 'log', filt = "median, movingAverage", skip = 0, printLast = True, show = False)

	T = 100

	ag = RandomAction(mu = 0, sigma = 0.3)

	exp.run_experiments_multiple(Setting4(), {"timesteps" : T},\
						 [OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering()
						 ],\
						 [{'timesteps': T, 'max_k' : 50, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 1, 'out_dim': 1, 'R_m': 7.0}],\
						 ['Reference',\
						 'OnlineWaveFiltering_10_4_1',\
						 'OnlineWaveFiltering_10_3_1',\
						 'OnlineWaveFiltering_10_2_1',\
						 'OnlineWaveFiltering_10_1_1',\
						 'OnlineWaveFiltering_10_0_1',\
						 'OnlineWaveFiltering_30_4_1',\
						 'OnlineWaveFiltering_30_3_1',\
						 'OnlineWaveFiltering_30_2_1',\
						 'OnlineWaveFiltering_30_1_1',\
						 'OnlineWaveFiltering_30_0_1',\
						 'OnlineWaveFiltering_50_4_1',\
						 'OnlineWaveFiltering_50_3_1',\
						 'OnlineWaveFiltering_50_2_1',\
						 'OnlineWaveFiltering_50_1_1',\
						 'OnlineWaveFiltering_50_0_1',\
						 'OnlineWaveFiltering_10_4_3',\
						 'OnlineWaveFiltering_10_3_3',\
						 'OnlineWaveFiltering_10_2_3',\
						 'OnlineWaveFiltering_10_1_3',\
						 'OnlineWaveFiltering_10_0_3',\
						 'OnlineWaveFiltering_30_4_3',\
						 'OnlineWaveFiltering_30_3_3',\
						 'OnlineWaveFiltering_30_2_3',\
						 'OnlineWaveFiltering_30_1_3',\
						 'OnlineWaveFiltering_30_0_3',\
						 'OnlineWaveFiltering_50_4_3',\
						 'OnlineWaveFiltering_50_3_3',\
						 'OnlineWaveFiltering_50_2_3',\
						 'OnlineWaveFiltering_50_1_3',\
						 'OnlineWaveFiltering_50_0_3',\
						 'OnlineWaveFiltering_10_4_5',\
						 'OnlineWaveFiltering_10_3_5',\
						 'OnlineWaveFiltering_10_2_5',\
						 'OnlineWaveFiltering_10_1_5',\
						 'OnlineWaveFiltering_10_0_5',\
						 'OnlineWaveFiltering_30_4_5',\
						 'OnlineWaveFiltering_30_3_5',\
						 'OnlineWaveFiltering_30_2_5',\
						 'OnlineWaveFiltering_30_1_5',\
						 'OnlineWaveFiltering_30_0_5',\
						 'OnlineWaveFiltering_50_4_5',\
						 'OnlineWaveFiltering_50_3_5',\
						 'OnlineWaveFiltering_50_2_5',\
						 'OnlineWaveFiltering_50_1_5',\
						 'OnlineWaveFiltering_50_0_5',\
						 'OnlineWaveFiltering_10_4_7',\
						 'OnlineWaveFiltering_10_3_7',\
						 'OnlineWaveFiltering_10_2_7',\
						 'OnlineWaveFiltering_10_1_7',\
						 'OnlineWaveFiltering_10_0_7',\
						 'OnlineWaveFiltering_30_4_7',\
						 'OnlineWaveFiltering_30_3_7',\
						 'OnlineWaveFiltering_30_2_7',\
						 'OnlineWaveFiltering_30_1_7',\
						 'OnlineWaveFiltering_30_0_7',\
						 'OnlineWaveFiltering_50_4_7',\
						 'OnlineWaveFiltering_50_3_7',\
						 'OnlineWaveFiltering_50_2_7',\
						 'OnlineWaveFiltering_50_1_7',\
						 'OnlineWaveFiltering_50_0_7'],
						 n_runs = 5,\
						 plotter = plotting,\
						 action_generator = ag,\
						 verbose = False)

def tune_exp_5():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(yscale = 'log', filt = "median, movingAverage", skip = 0, printLast = True, show = False)

	A = np.diag([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	B = np.eye(10)
	C = np.random.normal(size = (10, 10)) * 0.3
	D = np.zeros((10, 10))

	T = 100

	ag = BlockAction(prob_repeat = 0.8, sigma = 0.3)

	
	exp.run_experiments_multiple(LDS(), {'action_dim': 10, 'hidden_dim': 10, 'out_dim': 10, 'partially_observable': True,\
								 'system_params': {'A': A, 'B' : B, 'C': C, 'D': D}},\
						 [OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering(),\
						  OnlineWaveFiltering()
						 ],\
						 [{'timesteps': T, 'max_k' : 50, 'action_dim': 10, 'out_dim': 10, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 1.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 3.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 5.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 10, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 30, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-4, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-3, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-2, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1e-1, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0},\
						  {'timesteps': T, 'k' : 50, 'lr': 1.0, 'action_dim': 10, 'out_dim': 10, 'R_m': 7.0}],\
						 ['Reference',\
						 'OnlineWaveFiltering_10_4_1',\
						 'OnlineWaveFiltering_10_3_1',\
						 'OnlineWaveFiltering_10_2_1',\
						 'OnlineWaveFiltering_10_1_1',\
						 'OnlineWaveFiltering_10_0_1',\
						 'OnlineWaveFiltering_30_4_1',\
						 'OnlineWaveFiltering_30_3_1',\
						 'OnlineWaveFiltering_30_2_1',\
						 'OnlineWaveFiltering_30_1_1',\
						 'OnlineWaveFiltering_30_0_1',\
						 'OnlineWaveFiltering_50_4_1',\
						 'OnlineWaveFiltering_50_3_1',\
						 'OnlineWaveFiltering_50_2_1',\
						 'OnlineWaveFiltering_50_1_1',\
						 'OnlineWaveFiltering_50_0_1',\
						 'OnlineWaveFiltering_10_4_3',\
						 'OnlineWaveFiltering_10_3_3',\
						 'OnlineWaveFiltering_10_2_3',\
						 'OnlineWaveFiltering_10_1_3',\
						 'OnlineWaveFiltering_10_0_3',\
						 'OnlineWaveFiltering_30_4_3',\
						 'OnlineWaveFiltering_30_3_3',\
						 'OnlineWaveFiltering_30_2_3',\
						 'OnlineWaveFiltering_30_1_3',\
						 'OnlineWaveFiltering_30_0_3',\
						 'OnlineWaveFiltering_50_4_3',\
						 'OnlineWaveFiltering_50_3_3',\
						 'OnlineWaveFiltering_50_2_3',\
						 'OnlineWaveFiltering_50_1_3',\
						 'OnlineWaveFiltering_50_0_3',\
						 'OnlineWaveFiltering_10_4_5',\
						 'OnlineWaveFiltering_10_3_5',\
						 'OnlineWaveFiltering_10_2_5',\
						 'OnlineWaveFiltering_10_1_5',\
						 'OnlineWaveFiltering_10_0_5',\
						 'OnlineWaveFiltering_30_4_5',\
						 'OnlineWaveFiltering_30_3_5',\
						 'OnlineWaveFiltering_30_2_5',\
						 'OnlineWaveFiltering_30_1_5',\
						 'OnlineWaveFiltering_30_0_5',\
						 'OnlineWaveFiltering_50_4_5',\
						 'OnlineWaveFiltering_50_3_5',\
						 'OnlineWaveFiltering_50_2_5',\
						 'OnlineWaveFiltering_50_1_5',\
						 'OnlineWaveFiltering_50_0_5',\
						 'OnlineWaveFiltering_10_4_7',\
						 'OnlineWaveFiltering_10_3_7',\
						 'OnlineWaveFiltering_10_2_7',\
						 'OnlineWaveFiltering_10_1_7',\
						 'OnlineWaveFiltering_10_0_7',\
						 'OnlineWaveFiltering_30_4_7',\
						 'OnlineWaveFiltering_30_3_7',\
						 'OnlineWaveFiltering_30_2_7',\
						 'OnlineWaveFiltering_30_1_7',\
						 'OnlineWaveFiltering_30_0_7',\
						 'OnlineWaveFiltering_50_4_7',\
						 'OnlineWaveFiltering_50_3_7',\
						 'OnlineWaveFiltering_50_2_7',\
						 'OnlineWaveFiltering_50_1_7',\
						 'OnlineWaveFiltering_50_0_7'],
						 n_runs = 5,\
						 plotter = plotting,\
						 action_generator = ag,\
						 verbose = False)

if __name__ == '__main__':
	tune_exp_1()
	tune_exp_2()
	tune_exp_3()
	tune_exp_4()
	tune_exp_5()