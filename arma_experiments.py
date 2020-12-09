import numpy as np

from predictors.onlineWaveFiltering import OnlineWaveFiltering
from predictors.consistency import Consistency
from predictors.kalmanFilter import KalmanFilter
from predictors.onlineWaveFilteringParameterFree import OnlineWaveFilteringParameterFree
import matplotlib.pyplot as plt

#optimizers
from predictors.optimizers.ftrl import FTRL
from predictors.optimizers.hedge import Hedge
from predictors.armaAutoregressor import ARMA
from predictors.arimaAutoregressor import ARIMA

from predictors.var import VAR
from predictors.ecVarmaOgd import ECVARMAOGD

from environments.SP500 import SP500
from environments.room_temperature import RoomTemperature

from utils.plotter import Plotter
from utils.loss import quad_loss
from utils.experiment import Experiment
from utils.action_generation import *

import datetime

from predictors.optimizers.RealOGD import RealOGD

def sp500():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(filt = "movingAverage", skip = 100)

	exp.run_experiments_multiple(SP500(), {},\
						 [ARMA(),\
						  ARIMA(),\
						  OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFilteringParameterFree()
						  ],\
						 [{'p' : 64, 'optimizer': RealOGD(hyperparameters={'lr':10.0})},\
						  {'p' : 16, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':10.0})},\
						  {'max_k' : 20, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'max_k' : 50, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'max_k' : 100, 'action_dim': 1, 'out_dim': 1, 'opt': Hedge(), 'optForSubPredictors': FTRL()}
						  ],\
						 ['ARMA_OGD',\
						  'ARIMA_OGD',\
						  'OnlineWaveFilteringParameterFree20',\
						  'OnlineWaveFilteringParameterFree50',\
						  'OnlineWaveFilteringParameterFree100'
						  ],\
						 n_runs = 20,\
						 plotter = plotting,\
						 verbose = True,
						 action_generator = ProblemBasedAction(SP500()))


def roomTemperature():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(filt = "median, movingAverage", skip = 500)

	dimension = 20

	exp.run_experiments_multiple(RoomTemperature(), {},\
						 [VAR(),\
						  ECVARMAOGD(),\
						  OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFilteringParameterFree(),\
						  OnlineWaveFilteringParameterFree()
						  ],\
						 [{'p' : 8, 'dim': dimension},\
						  {'p' : 8, 'dim': dimension, 'lr': 1},\
						  {'max_k' : 10, 'action_dim': dimension, 'out_dim': dimension, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'max_k' : 30, 'action_dim': dimension, 'out_dim': dimension, 'opt': Hedge(), 'optForSubPredictors': FTRL()},\
						  {'max_k' : 50, 'action_dim': dimension, 'out_dim': dimension, 'opt': Hedge(), 'optForSubPredictors': FTRL()}
						  ],\
						 ['VAR_8',\
						  'ECVARMA_OGD16',\
						  'OnlineWaveFilteringParameterFree10',\
						  'OnlineWaveFilteringParameterFree30',\
						  'OnlineWaveFilteringParameterFree50'
						  ],\
						 n_runs = 20,\
						 plotter = plotting,\
						 verbose = True,
						 action_generator = ProblemBasedAction(RoomTemperature()))



if __name__ == '__main__':
	sp500()
	roomTemperature()
