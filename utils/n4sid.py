import numpy as np

# N4SID algorithm implementation for identification of the space following page 52 of the overscheer book.

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

def n4sid(x, y, order, k):
	"""
	x contains one input in each column
	y contains one output in each column
	"""

	m = x.shape[0]
	nx = x.shape[1]
	l = y.shape[0]
	ny = y.shape[1]

	# Determinar o numero de columns para as matrizes de Hankel
	N = ny - 2 * order + 1

	#Create block hankel matrices from in and output
	X = blockHankel(x, 2*order, N)
	Y = blockHankel(y, 2*order, N)
	
	Xp = X[:, 0 : k*m]#X[:, 1 : k*m]
	Xf = X[:, k*m : 2*k*m]#X[:, k*m+1 : 2*k*m]
	Yp = Y[:, 0:k*l]#Y[:, 1:k*l]
	Yf = Y[:, k*l:2*k*l]#Y[:, k*l+1:2*k*l]
	km = Xp.shape[0]#size(Xp,1);
	kl = Yp.shape[0]#size(Yp,1);
	Wp = np.concatenate((Xp, Xp), axis = 1)#[Xp;Xp];
	# *********** ALGORITMO ***********
	#Passo 1
	#decomposicao LQ
	Q, L = np.linalg.qr(np.concatenate((Xf, Xp, Yp, Yf), axis = 1).T)#np.linalg.qr([Xf;Xp;Yp;Yf].T);
	Q = Q.T
	L = L.T

	L11 = L[0:km, 0:km]#L[1:km,1:km]
	L21 = L[km:2*km, 0:km]#L[km+1:2*km,1:km]
	L22 = L[km:2*km, km+1:2*km]#L[km+1:2*km,km+1:2*km]
	L31 = L[2*km:2*km+kl, 0:km]#L[2*km+1:2*km+kl,1:km]
	L32 = L[2*km:2*km+kl, km:2*km]#L[2*km+1:2*km+kl,km+1:2*km]
	L33 = L[2*km:2*km+kl, 2*km:2*km+kl]#L[2*km+1:2*km+kl,2*km+1:2*km+kl]
	L41 = L[2*km+kl:2*km+2*kl, 0:km]#L[2*km+kl+1:2*km+2*kl,1:km]
	L42 = L[2*km+kl:2*km+2*kl, km:2*km]#L[2*km+kl+1:2*km+2*kl,km+1:2*km]
	L43 = L[2*km+kl:2*km+2*kl, 2*km:2*km+kl]#L[2*km+kl+1:2*km+2*kl,2*km+1:2*km+kl]
	L44 = L[2*km+kl:2*km+2*kl, 2*km+kl:2*km+2*kl]#L[2*km+kl+1:2*km+2*kl,2*km+kl+1:2*km+2*kl]
	R11 = L11

	print(L32)
	print(L33)
	R21 = np.concatenate((L21, L32), axis = 1)#[L21;L31];
	R22 = np.concatenate((np.concatenate((L22, np.zeros(km,kl)), axis = 0), np.concatenate((L32, L33), axis = 0)), axis = 1)#[L22 zeros(km,kl); L32 L33];
	R31 = L41
	R32 = np.concatenate((L42, L43), axis = 0)#[L42 L43]

	xi = R32.dot(np.linalg.pinv(R22).dot(Wp))
	#Passo 2
	XX, SS, VV = np.linalg.svd(xi)
	ss = np.diag(SS)
	n = np.argwhere(np.cumsum(ss) > 0.85 * np.sum(ss))[0]
	# n=4;
	# hold off
	# figure(1)
	# title('Valores singulares');
	# xlabel('Ordem');
	# plot(ss)
	# pause;
	# figure(2)
	# title('Valores singulares');
	# xlabel('Ordem');
	# bar(ss)
	# n = input(' Ordem do sistema ? ');
	# while isempty(n)
	# n = input(' Ordem do sistema ? ');
	# 
	X1 = XX[:, 1:order]
	S1 = SS[1:order, 1:order]
	V1 = VV[1:order, :]
	#Matrizes A e C
	Ok = X1.dot(np.linalg.sqrtm(S1))
	C = Ok[1:l, 1:order]
	A = np.linalg.pinv(Ok[1:l*(k-1), 1:n]).dot(Ok[l+1:l*k, 1:order])

	#Passo 3
	#Matrizes B e D
	TOEP = R31 - R32.dot(np.linalg.pinv(R22)).dot(R21).dot(np.linalg.pinv(R11))
	G = TOEP[:,1:m]
	G0 = G[1:l,:]
	G1 = G[l+1:2*l,:]
	G2 = G[2*l+1:3*l,:]
	G3 = G[3*l+1:4*l,:]
	G4 = G[4*l+1:5*l,:]
	D = G0
	Hk = np.concatenate((np.concatenate((G1, G2), axis = 0), np.concatenate((G2, G3), axis = 0), np.concatenate((G3, G4), axis = 0)), axis = 1)#[G1 G2;G2 G3;G3 G4];
	Ok1 = Ok[1:3*l,:]
	Ck = np.linalg.pinv(Ok1).dot(Hk)
	B = Ck[:,1:m]

	return A, B, C, D
