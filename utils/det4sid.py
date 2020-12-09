import numpy as np

#   Deterministic subspace identification (Algorithm 1)
#
#           [A,B,C,D] = det_stat(y,u,i);
# 
#   Inputs:
#           y: matrix of measured outputs
#           u: matrix of measured inputs
#           i: number of block rows in Hankel matrices 
#              (i * #outputs) is the max. order that can be estimated 
#              Typically: i = 2 * (max order)/(#outputs)
#           
#   Outputs:
#           A,B,C,D: deterministic state space system
#           
#                  x_{k+1) = A x_k + B u_k        
#                    y_k   = C x_k + D u_k
#
#   Optional:
#
#           [A,B,C,D,AUX,ss] = det_stat(y,u,i,n,AUX,sil);
#   
#           n:    optional order estimate (default [])
#           AUX:  optional auxilary variable to increase speed (default [])
#           ss:   column vector with singular values
#           sil:  when equal to 1 no text output is generated
#           
#   Example:
#   
#           [A,B,C,D,AUX] = det_stat(y,u,10,2);
#           for k=3:6
#              [A,B,C,D] = det_stat(y,u,10,k,AUX);
#           end

def blockHankel(x, nrows, ncols):
	"""
	x contains one input in each column
	"""

	assert(x.shape[1] >= nrows)

	m = x.shape[0]
	result = np.zeros((m * nrows, ncols))
	for col in range(ncols):
		for row in range(nrows):
			result[m * row : m * (row + 1), col] = x[:, col + row]

	return result


#function [A,B,C,D,AUX,ss] = det_stat(y,u,i,n,AUXin,sil);
def det4sid(u, y, i, n = [], sil = 0):

	# Weighting is always empty
	W = [];

	# Turn the data into row vectors and check
	m = u.shape[0]
	nu = u.shape[1]
	l = y.shape[0]
	ny = y.shape[1]

	assert(i >= 0)
	assert(l >= 0)
	assert(m >= 0)
	assert(nu == ny)
	#assert(nu - 2 * i + 1 >= 2 * l * i)

	# Determine the number of columns in Hankel matrices
	j = nu - 2 * i + 1

	# Check compatibility of AUXin
	#[AUXin,Wflag] = chkaux(AUXin, i, u(1,1), y(1,1), 1, W, sil); 
		
	# Compute the R factor
	U = blockHankel(u / np.sqrt(j), 2 * i, j) 		# Input block Hankel
	Y = blockHankel(y / np.sqrt(j), 2 * i, j) 		# Output block Hankel
	#mydisp(sil,'      Computing ... R factor');
	#R = np.triu(np.linalg.qr(np.concatenate((U, Y)).T)).T		#R = triu(qr([U;Y]'))'; 		# R factor
	_, R = np.linalg.qr(np.concatenate((U, Y)).T)		#R = triu(qr([U;Y]'))'; 		# R factor
	R = R.T
	R = R[:2*i*(m+l), :2*i*(m+l)] 	# Truncate

	#############################################################################
	#############################################################################
	#
	#                                  BEGIN ALGORITHM
	#
	#############################################################################
	#############################################################################



	# **************************************
	#               STEP 1 
	# **************************************

	mi2 = 2 * m * i

	# Set up some matrices
	Rf = R[(2*m+l)*i : 2*(m+l)*i, :] 	# Future outputs
	Rp = np.concatenate((R[:m*i, :], R[2*m*i : (2*m+l)*i, :])) # Past (inputs and) outputs
	Ru  = R[m*i:2*m*i, :mi2] 		# Future inputs

	# Perpendicular Future outputs 
	#Rfp = [Rf(:,1:mi2) - (Rf(:,1:mi2)/Ru)*Ru,Rf(:,mi2+1:2*(m+l)*i)];
	Rfp = np.concatenate((Rf[:, :mi2] - np.dot(Rf[:, :mi2], np.linalg.pinv(Ru)).dot(Ru), Rf[:, mi2:2*(m+l)*i]), axis = 1)
	# Perpendicular Past
	#Rpp = [Rp(:,1:mi2) - (Rp(:,1:mi2)/Ru)*Ru,Rp(:,mi2+1:2*(m+l)*i)];
	Rpp = np.concatenate((Rp[:, :mi2] - np.dot(Rp[:, :mi2], np.linalg.pinv(Ru)).dot(Ru), Rp[:, mi2:2*(m+l)*i]), axis = 1)


	if ((np.linalg.norm(Rpp[:, (2*m+l) * i - 2 * l - 1 : (2*m+l)*i],'fro')) < 1e-10):
		#Ob  = (Rfp*pinv(Rpp')')*Rp; #altered
		Ob  = np.dot(np.dot(Rfp, np.linalg.pinv(Rpp.T)).T, Rp) 	# Oblique projection
	else:
		Ob = np.dot(np.dot(Rfp, np.linalg.pinv(Rpp)), Rp)


	# **************************************
	#               STEP 2 
	# **************************************

	# Compute the SVD
	U, S, V = np.linalg.svd(Ob)
	ss = np.diag(S)
	

	# **************************************
	#               STEP 3 
	# **************************************

	# Determine the order from the singular values
	if (n == []):
		n = 0

	n = i

	#U1 = U(:,1:n); 				# Determine U1
	U1 = U[:,:n] 				# Determine U1


	# **************************************
	#               STEP 4 
	# **************************************

	# Determine gam and gamm
	#gam  = np.dot(U1, np.diag(np.sqrt(ss[:n])))
	gam  = np.dot(U1, np.sqrt(ss[:n]))
	#gamm = gam[:l*(i-1), :]
	gamm = gam[:l*(i-1)]
	# And their pseudo inverses
	gam_inv  = np.linalg.pinv(gam)
	gamm_inv = np.linalg.pinv(gamm)



	# **************************************
	#               STEP 5 
	# **************************************

	Rf = R[(2*m+l)*i + l : 2*(m+l)*i, :] 	# Future outputs
	Rp = np.concatenate((R[:m * (i + 1), :], R[2*m*i : (2*m+l)*i + l, :])) # Past (inputs and) outputs
	Ru  = R[m*i + m:2*m*i, :mi2] 		# Future inputs
	# Perpendicular Future outputs 
	#Rfp = [Rf(:,1:mi2) - (Rf(:,1:mi2)/Ru)*Ru,Rf(:,mi2+1:2*(m+l)*i)]; 
	Rfp = np.concatenate((Rf[:, :mi2] - np.dot(Rf[:, :mi2], np.linalg.pinv(Ru)).dot(Ru), Rf[:, mi2:2*(m+l)*i]), axis = 1)
	# Perpendicular Past
	#Rpp = [Rp(:,1:mi2) - (Rp(:,1:mi2)/Ru)*Ru,Rp(:,mi2+1:2*(m+l)*i)];
	Rpp = np.concatenate((Rp[:, :mi2] - np.dot(Rp[:, :mi2], np.linalg.pinv(Ru)).dot(Ru), Rp[:, mi2:2*(m+l)*i]), axis = 1)

	if ((np.linalg.norm(Rpp[:, (2*m+l) * i - 2 * l - 1 : (2*m+l)*i],'fro')) < 1e-10):
		#Ob  = (Rfp*pinv(Rpp')')*Rp; #altered
		Obm  = np.dot(np.dot(Rfp, np.linalg.pinv(Rpp.T)).T, Rp) 	# Oblique projection
		#Obm  = np.dot(np.dot(Rfp, np.linalg.pinv(Rpp.T).T), Rp) 	# Oblique projection
	else:
		Obm = np.dot(np.dot(Rfp, np.linalg.pinv(Rpp)), Rp)

	# Determine the states Xi and Xip
	Xi  = np.dot(gam_inv, Ob)
	Xip = np.dot(gamm_inv, Obm)


	# **************************************
	#               STEP 6 
	# **************************************
	
	Rhs = np.concatenate((Xi, R[m*i:m*(i+1), :])) # Right hand side
	Lhs = np.concatenate((Xip, R[(2*m+l)*i:(2*m+l)*i+l, :])) # Left hand side

	# Solve least squares
	#sol = Lhs / Rhs
	sol = np.dot(Lhs, np.linalg.pinv(Rhs))

	# Extract the system matrices
	A = sol[:n, :n]
	B = sol[:n, n:n+m]
	C = sol[n:n+l, :n]
	D = sol[n:n+l, n:n+m]

	return A, B, C, D