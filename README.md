# Online Wave Filtering Parameter Free
This repository contains the code to reproduce the experiment results for my master thesis.

# Requirements

To run these experiments the following software needs to be installed:

- python (https://www.python.org/downloads/)
- scipy (https://www.scipy.org/install.html)
- numpy (https://numpy.org/install/)
- joblib (https://pypi.org/project/joblib/)
- cv2 (https://pypi.org/project/opencv-python/)
- jax (https://pypi.org/project/jax/)
- cvxopt (https://cvxopt.org/install/)
- statsmodel (https://www.statsmodels.org/stable/install.html)
- tigercontrol (https://github.com/MinRegret/TigerControl)
- tigerforecast (https://github.com/MinRegret/TigerForecast)

# Run experiments

This repository contains different python files. If you want to rerun any of my experiments you just need to run the corresponding python file.

These files can be used to rerun the experiments.
1. [artificial_experiments.py](artificial_experiments.py)
2. [videotracking.py](videotracking.py)
3. [arma_experiments.py](arma_experiments.py)

These files can be used to rerun our tuning process.
1. [artificial_experiments_tuning.py](artificial_experiments_tuning.py)
2. [arma_experiments_tuning.py](arma_experiments_tuning.py)


To reproduce the results in my thesis select the corresponding file and run:
```python filename.py```
If you do not want to run all experiments adjust the main method.

Keep in mind that the videotracking experiment requires a blue ball and a working webcam.