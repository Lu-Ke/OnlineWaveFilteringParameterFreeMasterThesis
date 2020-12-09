import numpy as np

def _e(A, B, C, D, Q, R, x, y, h_init):
	Ptt1 = [Q]
	et = [y[:, 0] - (C.dot(h_init) + D.dot(x[:, 0]))]#np.random.randn(y.shape[0])
	Kt = [Ptt1[-1].dot(C.T).dot(np.linalg.inv(C.dot(Ptt1[-1].dot(C.T)) + R))]#np.random.randn(x.shape[0], y.shape[0])
	htt1 = [A.dot(h_init) + B.dot(x[:, 0])]
	htt = [htt1[-1] + Kt[-1].dot(et[-1])]
	Ptt = [Ptt1[-1] - Kt[-1].dot(C).dot(Ptt1[-1])]

	#forward pass
	for i in range(1, y.shape[1]):
		htt1.append(A.dot(htt[-1]) + B.dot(x[:, i]))
		Ptt1.append(A.dot(Ptt[-1]).dot(A.T) + Q)
		et.append(y[:, i] - (C.dot(htt1[-1]) + D.dot(x[:, i])))
		Kt.append(Ptt1[-1].dot(C.T).dot(np.linalg.inv(C.dot(Ptt1[-1].dot(C.T)) + R)))
		htt.append(htt1[-1] + Kt[-1].dot(et[-1]))
		Ptt.append(Ptt1[-1] - Kt[-1].dot(C).dot(Ptt1[-1]))

	#print(htt)

	#backward pass
	ht = [htt[-1]]
	Pt = [Ptt[-1]]
	for i in range(y.shape[1] - 2, -1, -1):
		Pt1t = A.dot(Ptt[i]).dot(A.T) + Q
		G = Ptt[i].dot(A).dot(np.linalg.inv(Pt1t))
		ht.append(htt[i] + G.dot(ht[-1] - (A.dot(htt[i + 1]) + B.dot(x[:, i + 1]))))
		Pt.append(Ptt[i] + G.dot(Pt1t - Ptt[i + 1]).dot(G.T))

	ht.reverse()
	#Pt.reverse()
	ht = np.array(ht).T
	Pt = np.mean(np.array(Pt).T, axis = 2)
	Pt = np.where(np.logical_and(np.abs(Pt) < 10, np.abs(Pt) > 1e-5), R, np.random.randn(Pt.shape[0], Pt.shape[1]) * 0.3)
	
	"""
	print(A)
	print("-----------------------")
	print(B)
	print("-----------------------")
	print(C)
	print("-----------------------")
	print(D)
	print("-----------------------")
	print(Q)
	print("-----------------------")
	print(R)
	print("-----------------------")
	print(h_init)
	print("-----------------------")
	print(ht)
	print("-----------------------")
	print(Pt)
	print("==================================")
	"""

	return ht, Pt

def _m(x, y, h_init, ht, order):
	wt = np.concatenate((ht, x), axis = 0)
	#CD = (y.dot(wt.T)).dot(np.linalg.inv(wt.dot(wt.T)))
	#AB = (ht[:, 1:].dot(wt[:, :-1].T)).dot(np.linalg.inv(wt[:, :-1].dot(wt[:, :-1].T)))

	CD = (y.dot(wt.T)).dot(np.linalg.inv(wt.dot(wt.T) + np.eye(wt.shape[0]) * 1e-6))
	AB = (ht[:, 1:].dot(wt[:, :-1].T)).dot(np.linalg.inv(wt[:, :-1].dot(wt[:, :-1].T) + np.eye(wt.shape[0]) * 1e-6))

	AB = np.where(np.logical_not(np.isnan(AB)), AB, np.random.randn(AB.shape[0], AB.shape[1]) * 0.3)
	AB = np.where(np.logical_and(np.abs(AB) < 10, np.abs(AB) > 1e-5), AB, np.random.randn(AB.shape[0], AB.shape[1]) * 0.3)
	CD = np.where(np.logical_not(np.isnan(CD)), CD, np.random.randn(CD.shape[0], CD.shape[1]) * 0.3)
	CD = np.where(np.logical_and(np.abs(CD) < 10, np.abs(CD) > 1e-5), CD, np.random.randn(CD.shape[0], CD.shape[1]) * 0.3)

	R = 1.0 / y.shape[1] * (y.dot(y.T) - CD.dot(wt.dot(y.T)))
	Q = 1.0 / (y.shape[1] - 1.0) * (ht[:, 1:].dot(ht[:, 1:].T) - AB.dot(wt[:, :-1].dot(ht[:, 1:].T)))

	R = np.where(np.logical_not(np.isnan(R)), R, np.random.randn(R.shape[0], R.shape[1]) * 0.3)
	R = np.where(np.logical_and(np.abs(R) < 10, np.abs(R) > 1e-5), R, np.random.randn(R.shape[0], R.shape[1]) * 0.3)
	Q = np.where(np.logical_not(np.isnan(Q)), Q, np.random.randn(Q.shape[0], Q.shape[1]) * 0.3)
	Q = np.where(np.logical_and(np.logical_not(np.isnan(Q)), np.abs(Q) < 10, np.abs(Q) > 1e-5), Q, np.random.randn(Q.shape[0], Q.shape[1]) * 0.3)


	A = AB[:, :order]
	B = AB[:, order:]
	C = CD[:, :order]
	D = CD[:, order:]



	#print(R)
	#print("-----------------------")
	#print(Q)
	#print("=======================")



	#print(A)
	#print(B)
	#print(C)
	#print(D)
	#print(ht)
	"""
	print(wt)
	print(wt.dot(wt.T))
	print(np.linalg.inv(wt.dot(wt.T)))
	print("-----------------------")
	print(wt[:, :-1].dot(ht[:, 1:].T))
	print(wt[:, :-1])
	print(ht[:, 1:].T)
	print("=======================")
	"""

	return A, B, C, D, Q, R

def em(x, y, order, k = 100, A = None, B = None, C = None, D = None):
	if(A is None):
		A = np.random.randn(order, order) * 0.3
	if(B is None):
		B = np.random.randn(order, x.shape[0]) * 0.3
	if(C is None):
		C = np.random.randn(y.shape[0], order) * 0.3
	if(D is None):
		D = np.random.randn(y.shape[0], x.shape[0]) * 0.3

	h_init = np.zeros(order)#np.random.randn(order) * 0.1#

	Q = np.random.randn(order, order) * 0.3
	R = np.random.randn(y.shape[0], y.shape[0]) * 0.3

	for i in range(k):
		ht, P = _e(A, B, C, D, Q, R, x, y, h_init)
		newA, newB, newC, newD, newQ, newR = _m(x, y, h_init, ht, order)
		#print(np.sum(np.abs(newA - A)) + np.sum(np.abs(newB - B)) + np.sum(np.abs(newC - C)) +\
		#	np.sum(np.abs(newD - D)) + np.sum(np.abs(newQ - Q)) + np.sum(np.abs(newR - R)))

		if(np.sum(np.abs(newA - A)) + np.sum(np.abs(newB - B)) + np.sum(np.abs(newC - C)) +\
			np.sum(np.abs(newD - D)) + np.sum(np.abs(newQ - Q)) + np.sum(np.abs(newR - R)) < 0.001):
			print("EM reached a stable point.")
			A, B, C, D, Q, R = newA, newB, newC, newD, newQ, newR
			break
		else:
			if(np.sum(np.abs(newA - A)) + np.sum(np.abs(newB - B)) + np.sum(np.abs(newC - C)) +\
				np.sum(np.abs(newD - D)) + np.sum(np.abs(newQ - Q)) + np.sum(np.abs(newR - R)) > 1e10):
				# reset
				print("reset")
				A = np.random.randn(order, order) * 0.3
				B = np.random.randn(order, x.shape[0]) * 0.3
				C = np.random.randn(y.shape[0], order) * 0.3
				D = np.random.randn(y.shape[0], x.shape[0]) * 0.3
				Q = np.random.randn(order, order) * 0.3
				R = np.random.randn(y.shape[0], y.shape[0]) * 0.3
			else:
				A, B, C, D, Q, R = newA, newB, newC, newD, newQ, newR

	return A, B, C, D, P, Q, R, h_init