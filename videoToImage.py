import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np

dectX = []
dectY = []
myX = []
myY = []
kalX = []
kalY = []

with open("vid2/position_file.csv") as csvfile:  
	rows = csv.reader(csvfile, delimiter=',')
	for row in list(rows)[1:]:
		dectX.append(float(row[1]))
		dectY.append(float(row[2]))
		myX.append(float(row[3]))
		myY.append(float(row[4]))
		kalX.append(float(row[5]))
		kalY.append(float(row[6]))


start = 230
end = 285

dectX.pop(266)
dectY.pop(266)
dectX.pop(266)
dectY.pop(266)

x = np.array(dectX[start : end - 2])
y = np.array(dectY[start : end - 2])
y = y[x < 1000] - 50
x = x[x < 1000]

myX = np.array(myX[start : end])
myY = np.array(myY[start : end]) - 50
kalX = np.array(kalX[start : end])
kalY = np.array(kalY[start : end]) - 50

#dectX.pop(262)
#dectY.pop(262)
#dectX.pop(262)
#dectY.pop(262)
#dectX.pop(262)
#dectY.pop(262)

#plt.scatter(dectX[startDect : endDect], dectY[startDect : endDect], linewidth = 1, label = "True center", c = 'g')
plt.scatter(x, y, linewidth = 2, label = "True center", c = 'g')
plt.scatter(myX, myY, linewidth = 2, label = "OnlineWaveFilteringParameterFree", facecolors = 'none', edgecolors = 'r')
plt.scatter(kalX, kalY, linewidth = 2, label = "Kalman Filter", facecolors = 'none', edgecolors = 'b')

datafile = "vid2/frame245.jpg"
img = cv2.imread(datafile)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, zorder=0, extent=[0, 1145, 680, 0])


plt.legend(loc = 'upper right', framealpha = 1.0)
plt.show()
