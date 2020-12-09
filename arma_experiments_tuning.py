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

def sp500Tuning():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(filt = "movingAverage", skip = 100)
	exp.run_experiments_multiple(SP500(), {},\
						 [ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ARIMA(),\
						  ],\
						 [{'p' : 8, 'optimizer': RealOGD(hyperparameters={'lr':10.0})},\
						  {'p' : 8, 'optimizer': RealOGD(hyperparameters={'lr':1.0})},\
						  {'p' : 8, 'optimizer': RealOGD(hyperparameters={'lr':0.1})},\
						  {'p' : 8, 'optimizer': RealOGD(hyperparameters={'lr':0.01})},\
						  {'p' : 8, 'optimizer': RealOGD(hyperparameters={'lr':0.001})},\
						  {'p' : 8, 'optimizer': RealOGD(hyperparameters={'lr':0.0001})},\
						  {'p' : 16, 'optimizer': RealOGD(hyperparameters={'lr':10.0})},\
						  {'p' : 16, 'optimizer': RealOGD(hyperparameters={'lr':1.0})},\
						  {'p' : 16, 'optimizer': RealOGD(hyperparameters={'lr':0.1})},\
						  {'p' : 16, 'optimizer': RealOGD(hyperparameters={'lr':0.01})},\
						  {'p' : 16, 'optimizer': RealOGD(hyperparameters={'lr':0.001})},\
						  {'p' : 16, 'optimizer': RealOGD(hyperparameters={'lr':0.0001})},\
						  {'p' : 32, 'optimizer': RealOGD(hyperparameters={'lr':10.0})},\
						  {'p' : 32, 'optimizer': RealOGD(hyperparameters={'lr':1.0})},\
						  {'p' : 32, 'optimizer': RealOGD(hyperparameters={'lr':0.1})},\
						  {'p' : 32, 'optimizer': RealOGD(hyperparameters={'lr':0.01})},\
						  {'p' : 32, 'optimizer': RealOGD(hyperparameters={'lr':0.001})},\
						  {'p' : 32, 'optimizer': RealOGD(hyperparameters={'lr':0.0001})},\
						  {'p' : 64, 'optimizer': RealOGD(hyperparameters={'lr':10.0})},\
						  {'p' : 64, 'optimizer': RealOGD(hyperparameters={'lr':1.0})},\
						  {'p' : 64, 'optimizer': RealOGD(hyperparameters={'lr':0.1})},\
						  {'p' : 64, 'optimizer': RealOGD(hyperparameters={'lr':0.01})},\
						  {'p' : 64, 'optimizer': RealOGD(hyperparameters={'lr':0.001})},\
						  {'p' : 64, 'optimizer': RealOGD(hyperparameters={'lr':0.0001})},\
						  {'p' : 8, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':10.0})},\
						  {'p' : 8, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':1.0})},\
						  {'p' : 8, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.1})},\
						  {'p' : 8, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.01})},\
						  {'p' : 8, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.001})},\
						  {'p' : 8, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.0001})},\
						  {'p' : 16, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':10.0})},\
						  {'p' : 16, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':1.0})},\
						  {'p' : 16, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.1})},\
						  {'p' : 16, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.01})},\
						  {'p' : 16, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.001})},\
						  {'p' : 16, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.0001})},\
						  {'p' : 32, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':10.0})},\
						  {'p' : 32, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':1.0})},\
						  {'p' : 32, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.1})},\
						  {'p' : 32, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.01})},\
						  {'p' : 32, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.001})},\
						  {'p' : 32, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.0001})},\
						  {'p' : 64, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':10.0})},\
						  {'p' : 64, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':1.0})},\
						  {'p' : 64, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.1})},\
						  {'p' : 64, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.01})},\
						  {'p' : 64, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.001})},\
						  {'p' : 64, 'd' : 2, 'optimizer': RealOGD(hyperparameters={'lr':0.0001})}
						  ],\
						 ['ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD',\
						  'ARIMA_OGD'
						  ],\
						 n_runs = 5,\
						 plotter = plotting,\
						 verbose = True,
						 action_generator = ProblemBasedAction(SP500()))


def roomTemperatureTuning():
	exp = Experiment()
	plotting = Plotter()
	plotting.initialize(filt = "median, movingAverage", skip = 0, printLast = True)

	dimension = 20

	exp.run_experiments_multiple(RoomTemperature(), {},\
						 [VAR(),\
						  VAR(),\
						  VAR(),\
						  VAR(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD(),\
						  ECVARMAOGD()
						 ],\
						 [{'p' : 8, 'dim': dimension},\
						  {'p' : 16, 'dim': dimension},\
						  {'p' : 32, 'dim': dimension},\
						  {'p' : 64, 'dim': dimension},\
						  {'p' : 8, 'dim': dimension, 'lr': 1},\
						  {'p' : 16, 'dim': dimension, 'lr': 1},\
						  {'p' : 32, 'dim': dimension, 'lr': 1},\
						  {'p' : 64, 'dim': dimension, 'lr': 1},\
						  {'p' : 8, 'dim': dimension, 'lr': 0.1},\
						  {'p' : 16, 'dim': dimension, 'lr': 0.1},\
						  {'p' : 32, 'dim': dimension, 'lr': 0.1},\
						  {'p' : 64, 'dim': dimension, 'lr': 0.1},\
						  {'p' : 8, 'dim': dimension, 'lr': 0.01},\
						  {'p' : 16, 'dim': dimension, 'lr': 0.01},\
						  {'p' : 32, 'dim': dimension, 'lr': 0.01},\
						  {'p' : 64, 'dim': dimension, 'lr': 0.01},\
						  {'p' : 8, 'dim': dimension, 'lr': 0.001},\
						  {'p' : 16, 'dim': dimension, 'lr': 0.001},\
						  {'p' : 32, 'dim': dimension, 'lr': 0.001},\
						  {'p' : 64, 'dim': dimension, 'lr': 0.001}
						  ],\
						 ['VAR_8',\
						  'VAR_16',\
						  'VAR_32',\
						  'VAR_64',\
						  'ECVARMA_OGD8_1',\
						  'ECVARMA_OGD16_1',\
						  'ECVARMA_OGD32_1',\
						  'ECVARMA_OGD64_1',\
						  'ECVARMA_OGD8_01',\
						  'ECVARMA_OGD16_01',\
						  'ECVARMA_OGD32_01',\
						  'ECVARMA_OGD64_01',\
						  'ECVARMA_OGD8_001',\
						  'ECVARMA_OGD16_001',\
						  'ECVARMA_OGD32_001',\
						  'ECVARMA_OGD64_001',\
						  'ECVARMA_OGD8_0001',\
						  'ECVARMA_OGD16_0001',\
						  'ECVARMA_OGD32_0001',\
						  'ECVARMA_OGD64_0001'
						  ],\
						 n_runs = 5,\
						 plotter = plotting,\
						 verbose = True,
						 action_generator = ProblemBasedAction(RoomTemperature()))

if __name__ == '__main__':
	sp500Tuning()
	roomTemperatureTuning()