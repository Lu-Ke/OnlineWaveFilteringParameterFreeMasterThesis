import numpy as np

# Quadratic Loss
def quad_loss(x, u, Q = None, R = None):
	x_contrib = x.T @ x if Q is None else x.T @ Q @ x
	u_contrib = u.T @ u if R is None else u.T @ R @ u
	
	return np.sum(x_contrib + u_contrib)

# Policy Loss
def policy_loss(params, determine_action, w, look_back, env, cost_fn = quad_loss):
	"""
	Description: 
	Args:
	"""
	y = np.zeros((env.n, 1))
	for h in range(look_back, 0, -1):
		v = determine_action(params, y, w[:-h])
		y = env.dyn(y, v) + w[-h] 

	# Don't update state at the end    
	v = determine_action(params, y, w)
	return cost_fn(y, v) 

# Action Loss
def action_loss(actions, w, look_back, env, cost_fn = quad_loss):
	"""
	Description: 
	Args:
	"""
	y = np.zeros((env.n, 1))
	for h in range(look_back, 1, -1):
		y = env.dyn(y, actions[-h]) + w[-h-1]

	# Don't update state at the end    
	v = actions[-1]
	return cost_fn(y, v) 