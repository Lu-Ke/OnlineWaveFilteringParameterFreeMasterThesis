import numpy as np
import cv2
import csv
from predictors.kalmanFilter import KalmanFilter
from predictors.onlineWaveFilteringParameterFree import OnlineWaveFilteringParameterFree
from predictors.optimizers.hedge import Hedge
from predictors.optimizers.ftrl import FTRL

# >>>>> Color to be tracked
MIN_H_BLUE = 200
MAX_H_BLUE = 300
# <<<<< Color to be tracked


def trackKalman(predictor):
	# Camera frame
	frame = None

	# >>>> Kalman Filter
	stateSize = 6
	measSize = 4
	contrSize = 0

	kf = predictor

	state = np.zeros(stateSize)		# [x,y,v_x,v_y,w,h]
	meas = np.zeros(measSize)		# [z_x,z_y,z_w,z_h]

	# cv::Mat procNoise(stateSize, 1, type)
	# [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

	# Transition State Matrix A
	# Note: set dT at each processing step!
	# [ 1 0 dT 0  0 0 ]
	# [ 0 1 0  dT 0 0 ]
	# [ 0 0 1  0  0 0 ]
	# [ 0 0 0  1  0 0 ]
	# [ 0 0 0  0  1 0 ]
	# [ 0 0 0  0  0 1 ]
	#cv2.setIdentity(kf.transitionMatrix)

	# Measure Matrix H
	# [ 1 0 0 0 0 0 ]
	# [ 0 1 0 0 0 0 ]
	# [ 0 0 0 0 1 0 ]
	# [ 0 0 0 0 0 1 ]
	H = np.eye(stateSize)[[0, 1, 4, 5]]

	# Process Noise Covariance Matrix Q
	# [ Ex   0   0     0     0    0  ]
	# [ 0    Ey  0     0     0    0  ]
	# [ 0    0   Ev_x  0     0    0  ]
	# [ 0    0   0     Ev_y  0    0  ]
	# [ 0    0   0     0     Ew   0  ]
	# [ 0    0   0     0     0    Eh ]
	#cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2))
	processNoiseCov = np.zeros((stateSize, stateSize))
	processNoiseCov[0, 0] = 1e-2
	processNoiseCov[1, 1] = 1e-2
	processNoiseCov[2, 2] = 5.0
	processNoiseCov[3, 3] = 5.0
	processNoiseCov[4, 4] = 1e-2
	processNoiseCov[5, 5] = 1e-2

	# Measures Noise Covariance Matrix R
	#cv2.setIdentity(kf.measurementNoiseCov, 1e-1)	#TODO
	# <<<< Kalman Filter

	#kf.initialize({'h': state, 'A' : np.eye(6), 'B': np.zeros(6), 'C': np.eye(6), 'P': np.eye(6), 'Q': np.zeros(6), 'R': processNoiseCov})
	kf.initialize({'h': state, 'A' : np.eye(6), 'B': np.zeros(6), 'C': np.eye(6), 'P': np.eye(6), 'Q': processNoiseCov, 'R': np.eye(6)})

	# Camera Index
	idx = -1

	# Camera Capture
	cap = cv2.VideoCapture(idx)
	#succ = cap.open(idx)

	succ = True

	# >>>>> Camera Settings
	if (not succ):
		print("Webcam not connected.\nPlease verify")
		exit(1)

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

	print("\nHit 'q' to exit...\n")

	ch = 0

	ticks = 0
	found = False

	notFoundCount = 0

	# >>>>> Main loop
	while (ch != 'q'and ch != 'Q'):
		precTick = ticks
		ticks = cv2.getTickCount()

		dT = (1.0 * (ticks - precTick)) / cv2.getTickFrequency()

		# Frame acquisition
		_, frame = cap.read()

		#cv2.imshow("Original", frame)

		res = frame.copy()

		if (found):
			# >>>> Matrix A
			kf.A[0, 2] = dT
			kf.A[1, 3] = dT
			# <<<< Matrix A

			print("dT: " + str(dT))

			state = kf.predict(0)
			print("State post: " + str(state))

			rectWidth = int(state[4])
			rectHeight = int(state[5])
			rectX = int(state[0] - rectWidth // 2.0)
			rectY = int(state[1] - rectHeight // 2.0)

			centerX = int(state[0])
			centerY = int(state[1])
			cv2.circle(res, (centerX, centerY), 2, (0,0,255))

			cv2.rectangle(res, (rectX, rectY), (rectX + rectWidth, rectY + rectHeight), (0,0,255), 2)

		# >>>>> Noise smoothing
		blur = cv2.GaussianBlur(frame, (5, 5), 3.0, 3.0)
		# <<<<< Noise smoothing

		# >>>>> HSV conversion
		frmHsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		# <<<<< HSV conversion

		# >>>>> Color Thresholding
		# Note: change parameters for different colors
		rangeRes = cv2.inRange(frmHsv, (MIN_H_BLUE / 2, 100, 80), (MAX_H_BLUE / 2, 255, 255))
		# <<<<< Color Thresholding

		# >>>>> Improving the result
		#TODO I removed something here
		rangeRes = cv2.erode(rangeRes, np.eye(1), iterations = 2)
		rangeRes = cv2.dilate(rangeRes, np.eye(1), iterations = 2)
		# <<<<< Improving the result


		# Thresholding viewing
		cv2.imshow("Threshold", rangeRes)

		# >>>>> Contours detection
		contours, hierarchy = cv2.findContours(rangeRes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		# <<<<< Contours detection

		# >>>>> Filtering
		balls = []
		ballsBox = []
		for i in range(len(contours)):
			bBox = cv2.boundingRect(contours[i])

			ratio = (1.0 * bBox[2]) / bBox[3]
			if (ratio > 1.0):
				ratio = 1.0 / ratio

			# Searching for a bBox almost square
			if (ratio > 0.75 and bBox[2] * bBox[3] >= 400):
				balls.append(contours[i])
				ballsBox.append(bBox)
		# <<<<< Filtering

		print("Balls found: " + str(len(ballsBox)))
		#assume: (x, y, width, height)

		# >>>>> Detection result
		for i in range(len(balls)):
			cv2.drawContours(res, balls, i, (20, 150, 20), 1)
			cv2.rectangle(res, ballsBox[i], (0, 255, 0), 2)

			centerX = int(ballsBox[i][0] + ballsBox[i][2] // 2)
			centerY = int(ballsBox[i][1] + ballsBox[i][3] // 2)
			cv2.circle(res, (centerX, centerY), 2, (20, 150, 20))

			#stringstream sstr
			#sstr << "(" << center.x << "," << center.y << ")"
			#cv::putText(res, sstr.str(),
			#			cv::Point(center.x + 3, center.y - 3),
			#			cv::FONT_HERSHEY_SIMPLEX, 0.5, (20,150,20), 2)
		# <<<<< Detection result

		# >>>>> Kalman Update
		if (len(balls) == 0):
			notFoundCount += 1
			print("notFoundCount: " + str(notFoundCount))
			if( notFoundCount >= 100 ):
				found = False
		else:
			notFoundCount = 0

			meas[0] = ballsBox[0][0] + ballsBox[0][2] / 2
			meas[1] = ballsBox[0][1] + ballsBox[0][3] / 2
			meas[2] = ballsBox[0][2]
			meas[3] = ballsBox[0][3]

			if (not found): # First detection!
				# >>>> Initialization
				kf.P = np.eye(stateSize)

				state[0] = meas[0]
				state[1] = meas[1]
				state[2] = 0
				state[3] = 0
				state[4] = meas[2]
				state[5] = meas[3]
				# <<<< Initialization

				kf.h = state
				
				found = True
			else:
				up = np.array([meas[0], meas[1], 0.0, 0.0, meas[2], meas[3]])
				#kf.update_parameters(meas) # Correct
				kf.update_parameters(up)

			print("Measure matrix: " + str(meas))

		# <<<<< Kalman Update

		# Final result
		cv2.imshow("Tracking", res)

		# User key
		ch = cv2.waitKey(1)

#=======================================================================================================================================#
#=======================================================================================================================================#
#=======================================================================================================================================#
#=======================================================================================================================================#
#=======================================================================================================================================#




def track():
	# Camera frame
	frame = None

	# >>>> Kalman Filter
	stateSize = 4
	measSize = 4
	contrSize = 0

	kf = OnlineWaveFilteringParameterFree()

	kf.initialize({'timesteps': 20000, 'max_k' : 30, 'action_dim': 2 * measSize, 'out_dim': stateSize, 'opt': FTRL(), 'optForSubPredictors': FTRL()})

	state = np.zeros(stateSize)		# [x,y,v_x,v_y,w,h]
	meas = np.zeros(measSize)		# [z_x,z_y,z_w,z_h]
	lastMeas = meas
	lastLastMeas = lastMeas

	# Camera Index
	idx = -1

	# Camera Capture
	cap = cv2.VideoCapture(idx)
	#succ = cap.open(idx)

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

	print("\nHit 'q' to exit...\n")

	ch = 0

	ticks = 0
	found = False

	notFoundCount = 0

	# >>>>> Main loop
	while (ch != 'q'and ch != 'Q'):
		precTick = ticks
		ticks = cv2.getTickCount()

		dT = (1.0 * (ticks - precTick)) / cv2.getTickFrequency()

		# Frame acquisition
		_, frame = cap.read()

		#cv2.imshow("Original", frame)

		res = frame.copy()

		if (found):
			#print(np.concatenate((meas, lastMeas)).shape)
			#state = kf.predict(np.concatenate((meas, lastMeas, lastLastMeas)))
			state = kf.predict(np.concatenate((meas, lastMeas)))
			#state = kf.predict(meas)
			print("State post: " + str(state))

			rectWidth = int(state[2])
			rectHeight = int(state[3])
			rectX = int(state[0] - rectWidth / 2.0)
			rectY = int(state[1] - rectHeight / 2.0)

			centerX = int(state[0])
			centerY = int(state[1])
			cv2.circle(res, (centerX, centerY), 2, (0, 0, 255))

			upper = (rectX, rectY)
			lower = (rectX + rectWidth, rectY + rectHeight)

			try:
				cv2.rectangle(res, upper, lower, (0, 0, 255), 2)
				cv2.putText(res, "(" + str(centerX) + ", " + str(centerY) +")", (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
			except TypeError as e:
				print(str(e))

		else:
			#print(np.concatenate((meas, lastMeas)).shape)
			#state = kf.predict(np.concatenate((meas, lastMeas, lastLastMeas)))
			state = kf.predict(np.concatenate((meas, lastMeas)))
			#state = kf.predict(meas)
			lastLastMeas = lastMeas
			lastMeas = meas
			meas = state
			print("State post: " + str(state))

			rectWidth = int(state[2])
			rectHeight = int(state[3])
			rectX = int(state[0] - rectWidth / 2.0)
			rectY = int(state[1] - rectHeight / 2.0)

			centerX = int(state[0])
			centerY = int(state[1])
			cv2.circle(res, (centerX, centerY), 2, (0, 0, 255))

			upper = (rectX, rectY)
			lower = (rectX + rectWidth, rectY + rectHeight)

			try:
				cv2.rectangle(res, upper, lower, (0, 0, 255), 2)
				cv2.putText(res, "(" + str(centerX) + ", " + str(centerY) +")", (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
			except TypeError as e:
				print(str(e))


		# >>>>> Noise smoothing
		blur = cv2.medianBlur(frame, 5)#cv2.GaussianBlur(frame, (5, 5), 3.0, 3.0)
		# <<<<< Noise smoothing

		# >>>>> HSV conversion
		frmHsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		# <<<<< HSV conversion

		# >>>>> Color Thresholding
		# Note: change parameters for different colors
		rangeRes = cv2.inRange(frmHsv, (MIN_H_BLUE / 2, 100, 80), (MAX_H_BLUE / 2, 255, 255))
		# <<<<< Color Thresholding

		# >>>>> Improving the result
		#TODO I removed something here
		rangeRes = cv2.erode(rangeRes, np.eye(1), iterations = 5)
		rangeRes = cv2.dilate(rangeRes, np.eye(1), iterations = 5)
		# <<<<< Improving the result


		# Thresholding viewing
		cv2.imshow("Threshold", rangeRes)

		# >>>>> Contours detection
		contours, hierarchy = cv2.findContours(rangeRes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		# <<<<< Contours detection

		# >>>>> Filtering
		balls = []
		ballsBox = []
		for i in range(len(contours)):
			bBox = cv2.boundingRect(contours[i])

			ratio = (1.0 * bBox[2]) / bBox[3]
			if (ratio > 1.0):
				ratio = 1.0 / ratio

			# Searching for a bBox almost square
			if (ratio > 0.75 and bBox[2] * bBox[3] >= 400):
				balls.append(contours[i])
				ballsBox.append(bBox)
		# <<<<< Filtering

		print("Balls found: " + str(len(ballsBox)))
		#assume: (x, y, width, height)

		# >>>>> Detection result
		for i in range(len(balls)):
			cv2.drawContours(res, balls, i, (20, 150, 20), 1)
			cv2.rectangle(res, ballsBox[i], (0, 255, 0), 2)

			centerX = int(ballsBox[i][0] + ballsBox[i][2] // 2)
			centerY = int(ballsBox[i][1] + ballsBox[i][3] // 2)
			cv2.circle(res, (centerX, centerY), 2, (20, 150, 20))

			#stringstream sstr
			#sstr << "(" << center.x << "," << center.y << ")"
			#cv::putText(res, sstr.str(),
			#			cv::Point(center.x + 3, center.y - 3),
			#			cv::FONT_HERSHEY_SIMPLEX, 0.5, (20,150,20), 2)
		# <<<<< Detection result

		# >>>>> Kalman Update
		if (len(balls) == 0):
			notFoundCount += 1
			print("notFoundCount: " + str(notFoundCount))
			if( notFoundCount >= 100 ):
				found = False
		else:
			notFoundCount = 0
			lastLastMeas = lastMeas
			lastMeas = meas

			meas[0] = ballsBox[0][0] + ballsBox[0][2] / 2
			meas[1] = ballsBox[0][1] + ballsBox[0][3] / 2
			meas[2] = ballsBox[0][2]
			meas[3] = ballsBox[0][3]

			if (not found): # First detection!
				state = meas

				found = True
			else:
				kf.update_parameters(meas) # Correct

			print("Measure matrix: " + str(meas))

		# Final result
		cv2.imshow("Tracking", res)

		# User key
		ch = cv2.waitKey(1)



def trackBoth():
	

	#with open('vid2/position_file.csv', mode='w') as file:
	#	writer = csv.writer(file, delimiter=',')
	#	writer.writerow(["count", "detectedX", "detectedY", "myX", "myY", "kalX", "kalY"])


	# Camera frame

	frame = None
	count = 0

	# >>>> Kalman Filter
	stateSize = 4
	measSize = 4
	contrSize = 0

	kf = OnlineWaveFilteringParameterFree()


	kf.initialize({'timesteps': 3000, 'max_k' : 30, 'action_dim': 1 * measSize, 'out_dim': stateSize, 'opt': FTRL(), 'optForSubPredictors': FTRL()})

	state = np.zeros(stateSize)		# [x,y,v_x,v_y,w,h]
	meas = np.zeros(measSize)		# [z_x,z_y,z_w,z_h]
	lastMeas = meas
	lastLastMeas = lastMeas


	kfOld = KalmanFilter()

	state2 = np.zeros(6)


	H = np.eye(6)[[0, 1, 4, 5]]

	# Process Noise Covariance Matrix Q
	# [ Ex   0   0     0     0    0  ]
	# [ 0    Ey  0     0     0    0  ]
	# [ 0    0   Ev_x  0     0    0  ]
	# [ 0    0   0     Ev_y  0    0  ]
	# [ 0    0   0     0     Ew   0  ]
	# [ 0    0   0     0     0    Eh ]
	#cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2))
	processNoiseCov = np.zeros((6, 6))
	processNoiseCov[0, 0] = 1e-2
	processNoiseCov[1, 1] = 1e-2
	processNoiseCov[2, 2] = 5.0
	processNoiseCov[3, 3] = 5.0
	processNoiseCov[4, 4] = 1e-2
	processNoiseCov[5, 5] = 1e-2

	# Measures Noise Covariance Matrix R
	#cv2.setIdentity(kf.measurementNoiseCov, 1e-1)	#TODO
	# <<<< Kalman Filter

	#kf.initialize({'h': state, 'A' : np.eye(6), 'B': np.zeros(6), 'C': np.eye(6), 'P': np.eye(6), 'Q': np.zeros(6), 'R': processNoiseCov})
	kfOld.initialize({'h': state2, 'A' : np.eye(6), 'B': np.zeros(6), 'C': np.eye(6), 'D': np.zeros(6), 'P': np.eye(6), 'Q': processNoiseCov, 'R': np.eye(6)})

	# Camera Index
	idx = -1

	# Camera Capture
	cap = cv2.VideoCapture(idx)
	#succ = cap.open(idx)

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

	print("\nHit 'q' to exit...\n")

	ch = 0

	ticks = 0
	found = False

	notFoundCount = 0

	detectedX = 0
	detectedY = 0
	myX = 0
	myY = 0
	kalX = 0
	kalY = 0

	# >>>>> Main loop
	while (ch != 'q'and ch != 'Q'):
		precTick = ticks
		ticks = cv2.getTickCount()

		dT = (1.0 * (ticks - precTick)) / cv2.getTickFrequency()

		# Frame acquisition
		_, frame = cap.read()

		#cv2.imshow("Original", frame)

		res = frame.copy()

		if (found):
			#print(np.concatenate((meas, lastMeas)).shape)
			#state = kf.predict(np.concatenate((meas, lastMeas, lastLastMeas)))
			#state = kf.predict(np.concatenate((meas, lastMeas)))
			state = kf.predict(meas)
			print("State post: " + str(state))

			rectWidth = int(state[2])
			rectHeight = int(state[3])
			rectX = int(state[0] - rectWidth / 2.0)
			rectY = int(state[1] - rectHeight / 2.0)

			centerX = int(state[0])
			centerY = int(state[1])
			cv2.circle(res, (centerX, centerY), 2, (0, 0, 255))

			upper = (rectX, rectY)
			lower = (rectX + rectWidth, rectY + rectHeight)
			myX = centerX
			myY = centerY

			try:
				cv2.rectangle(res, upper, lower, (0, 0, 255), 2)
				cv2.putText(res, "(" + str(centerX) + ", " + str(centerY) +")", (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
			except TypeError as e:
				print(str(e))

			#Kalman

			# >>>> Matrix A
			kfOld.A[0, 2] = dT
			kfOld.A[1, 3] = dT
			# <<<< Matrix A

			print("dT: " + str(dT))

			state2 = kfOld.predict(0)
			print("State post: " + str(state2))

			rectWidth = int(state2[4])
			rectHeight = int(state2[5])
			rectX = int(state2[0] - rectWidth // 2.0)
			rectY = int(state2[1] - rectHeight // 2.0)

			centerX = int(state2[0])
			centerY = int(state2[1])
			kalX = centerX
			kalY = centerY
			cv2.circle(res, (centerX, centerY), 2, (255,0,0))

			cv2.rectangle(res, (rectX, rectY), (rectX + rectWidth, rectY + rectHeight), (255,0,0), 2)
			cv2.putText(res, "(" + str(centerX) + ", " + str(centerY) +")", (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

		else:
			#print(np.concatenate((meas, lastMeas)).shape)
			#state = kf.predict(np.concatenate((meas, lastMeas, lastLastMeas)))
			#state = kf.predict(np.concatenate((meas, lastMeas)))
			state = kf.predict(meas)
			lastLastMeas = lastMeas
			lastMeas = meas
			meas = state
			print("State post: " + str(state))

			rectWidth = int(state[2])
			rectHeight = int(state[3])
			rectX = int(state[0] - rectWidth / 2.0)
			rectY = int(state[1] - rectHeight / 2.0)

			centerX = int(state[0])
			centerY = int(state[1])
			cv2.circle(res, (centerX, centerY), 2, (0, 0, 255))

			myX = centerX
			myY = centerY

			upper = (rectX, rectY)
			lower = (rectX + rectWidth, rectY + rectHeight)

			try:
				cv2.rectangle(res, upper, lower, (0, 0, 255), 2)
				cv2.putText(res, "(" + str(centerX) + ", " + str(centerY) +")", (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
			except TypeError as e:
				print(str(e))


		# >>>>> Noise smoothing
		blur = cv2.medianBlur(frame, 5)#cv2.GaussianBlur(frame, (5, 5), 3.0, 3.0)
		# <<<<< Noise smoothing

		# >>>>> HSV conversion
		frmHsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		# <<<<< HSV conversion

		# >>>>> Color Thresholding
		# Note: change parameters for different colors
		rangeRes = cv2.inRange(frmHsv, (MIN_H_BLUE / 2, 100, 80), (MAX_H_BLUE / 2, 255, 255))
		# <<<<< Color Thresholding

		# >>>>> Improving the result
		#TODO I removed something here
		rangeRes = cv2.erode(rangeRes, np.eye(1), iterations = 5)
		rangeRes = cv2.dilate(rangeRes, np.eye(1), iterations = 5)
		# <<<<< Improving the result


		# Thresholding viewing
		cv2.imshow("Threshold", rangeRes)

		# >>>>> Contours detection
		contours, hierarchy = cv2.findContours(rangeRes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		# <<<<< Contours detection

		# >>>>> Filtering
		balls = []
		ballsBox = []
		for i in range(len(contours)):
			bBox = cv2.boundingRect(contours[i])

			ratio = (1.0 * bBox[2]) / bBox[3]
			if (ratio > 1.0):
				ratio = 1.0 / ratio

			# Searching for a bBox almost square
			if (ratio > 0.75 and bBox[2] * bBox[3] >= 400):
				balls.append(contours[i])
				ballsBox.append(bBox)
		# <<<<< Filtering

		print("Balls found: " + str(len(ballsBox)))
		#assume: (x, y, width, height)

		# >>>>> Detection result
		for i in range(len(balls)):
			cv2.drawContours(res, balls, i, (20, 150, 20), 1)
			cv2.rectangle(res, ballsBox[i], (0, 255, 0), 2)

			centerX = int(ballsBox[i][0] + ballsBox[i][2] // 2)
			centerY = int(ballsBox[i][1] + ballsBox[i][3] // 2)
			cv2.circle(res, (centerX, centerY), 2, (20, 150, 20))
			cv2.putText(res, "(" + str(centerX) + ", " + str(centerY) +")", (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

			detectedX = centerX
			detectedY = centerY

			#stringstream sstr
			#sstr << "(" << center.x << "," << center.y << ")"
			#cv::putText(res, sstr.str(),
			#			cv::Point(center.x + 3, center.y - 3),
			#			cv::FONT_HERSHEY_SIMPLEX, 0.5, (20,150,20), 2)
		# <<<<< Detection result

		# >>>>> Kalman Update
		if (len(balls) == 0):
			notFoundCount += 1
			print("notFoundCount: " + str(notFoundCount))
			if( notFoundCount >= 100 ):
				found = False
		else:
			notFoundCount = 0
			lastLastMeas = lastMeas
			lastMeas = meas

			meas[0] = ballsBox[0][0] + ballsBox[0][2] / 2
			meas[1] = ballsBox[0][1] + ballsBox[0][3] / 2
			meas[2] = ballsBox[0][2]
			meas[3] = ballsBox[0][3]

				
			if (not found): # First detection!
				# >>>> Initialization

				state2[0] = meas[0]
				state2[1] = meas[1]
				state2[2] = 0
				state2[3] = 0
				state2[4] = meas[2]
				state2[5] = meas[3]
				# <<<< Initialization

				kfOld.h = state2
				kfOld.P = np.eye(6)

				state = meas

				found = True
			else:
				up = np.array([meas[0], meas[1], 0.0, 0.0, meas[2], meas[3]])
				#kf.update_parameters(meas) # Correct
				kfOld.update_parameters(up)
				kf.update_parameters(meas) # Correct

			print("Measure matrix: " + str(meas))

		# Final result
		cv2.imshow("Tracking", res)

		#cv2.imwrite("vid2/frame%d.jpg" % count, res)
		#with open('vid2/position_file.csv', mode='a') as file:
		#	writer = csv.writer(file, delimiter=',')
		#	writer.writerow([count, detectedX, detectedY, myX, myY, kalX, kalY])

		count += 1

		# User key
		ch = cv2.waitKey(1)



if __name__ == '__main__':
	#trackKalman(KalmanFilter())

	#track()
	trackBoth()