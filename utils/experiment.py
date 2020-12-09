from joblib import Parallel, delayed
import multiprocessing
import datetime

import numpy as np
#from ...predictors import Predictor
#from ...environments import Environment

from .action_generation import RandomAction

class Experiment(object):

	def __init__(self):
		pass

	def run_experiment(self, environment, env_params, predictor, predictor_params, action_generator = RandomAction(), verbose = False, id = 0):
		#clone for parallelization
		env = environment.clone()
		pred = predictor.clone()
		ag = action_generator.clone()

		#initialize
		env.initialize(env_params)
		if(not "timesteps" in predictor_params):
			predictor_params["timesteps"] = env.getTimeSteps()

		tsteps = predictor_params["timesteps"]
		pred.initialize(predictor_params)

		losses = []

		for t in range(tsteps):
			x = ag.next(env.get_action_dim())
			pred.predict(x)
			y = env.step(x)
			#print(str(x) + " " + str(y))
			losses.append(pred.update_parameters(y))
			if(verbose and t % (tsteps // 10) == 0):
			#if(verbose and t % (env.getTimeSteps() // 4) == 0):
				print(str((100.0 * t) / tsteps) + "% of run " + str(id) + " done at " + str(datetime.datetime.now()), flush = True)
		if(verbose):
			print("Run " + str(id) + " finished at " + str(datetime.datetime.now()), flush = True)

		return losses

	def run_experiments(self, environment, env_params, predictor, predictor_params, action_generator = RandomAction(), n_runs = 5, plotter = None, verbose = False):

		results = Parallel(n_jobs = max(7, multiprocessing.cpu_count() - 1))(delayed(self.run_experiment)(environment, env_params, predictor, predictor_params, action_generator, verbose, (i + 1)) for i in range(n_runs))

		valid_runs = n_runs
		losses = np.zeros(len(results[0]))
		for i in range(len(results)):
			if(np.any(np.isnan(np.array(results[i]))) or np.any(np.array(results[i]) > 1e10)):
				#print("Nope")
				valid_runs -= 1
				continue
			losses += np.array(results[i])

		if(valid_runs == 0):
			return self.run_experiments(environment, env_params, predictor, predictor_params, action_generator, n_runs, plotter, verbose)

		losses /= valid_runs

		if(plotter != None):
			plotter.plot([losses])

		return losses

	def run_experiments_multiple(self, environment, env_params, predictors, predictor_params, predictor_names, action_generator = RandomAction(), n_runs = 5, plotter = None, verbose = False):

		results = []

		for pred in range(len(predictors)):
			if(verbose):
				print("Starting experiment " + str(pred + 1) + " of " + str(len(predictors)), flush = True)
			results.append(self.run_experiments(environment, env_params, predictors[pred], predictor_params[pred], action_generator, n_runs, None, verbose))

		#print(np.array(results).shape)
		#exit()

		if(plotter != None):
			#print(results)
			plotter.plot(results, predictor_names)

		return results