import cv2
import numpy as np
import math
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from CustomParamsSimpleBlob import getMyBlobParams
from FileWriter import writeToFile

# Fetch learning set and train model with KNClassifier (tweak n_neighbors until best result)
mnist = fetch_mldata("MNIST original")
trainedModel = KNeighborsClassifier(n_neighbors=7, algorithm='brute').fit(mnist.data, mnist.target)


# File output
resultsString = "RA 68/2015 Masa Djurkovic\nfile\tsum"

# Iterate through 10 videos
for br in range(10):

	media = "video-" + str(br) + ".avi"
	capturing = cv2.VideoCapture(media)
	frameNum = 0
	capturing.set(1, frameNum)

	sumInVideo = 0


	while True:
	# Check hasFrames if it is end od video, and frame as image in video
	    hasFrames, frame = capturing.read()
	    frameNum += 1

	# Break while loop if end of video
	    if not hasFrames:
		break

	# For better performance take every 12th frame
	    if (frameNum % 12 == 0):

		# Flatten image color to gray (0-1, white-black)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Find edges/contours
	    	low_threshold = 50
	    	high_threshold = 150
	    	edges = cv2.Canny(gray, low_threshold, high_threshold)

		# Find big line below which number cross
	    	rho = 1
	    	theta = np.pi / 180
	    	threshold = 15
	    	min_line_length = 100
	    	max_line_gap = 10

	    	lines = cv2.HoughLinesP(edges, rho, theta, 100, threshold, min_line_length, max_line_gap)

		# HoughLines returns list of lines, sto take first element
		# Move points of line few pixels so we can get clear number instead one under line
		# Get lenght of line	    
	    	for x1,y1,x2,y2 in lines[0]:
			 x1=x1+10
			 x2=x2+10
			 y1=y1+10
			 y2=y2+10
		         length = math.sqrt(math.pow((x2-x1),2)+math.pow((y2-y1),2))

		
		# To repaint low bright pixels to black to remove white noise
	    	ret,noWhiteNoise = cv2.threshold(gray,160,255,cv2.THRESH_TOZERO)

		blobParams = getMyBlobParams()
		detector = cv2.SimpleBlobDetector_create(blobParams)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

		# Dilate bright spots 3 times to create blob for detection
		dilated = cv2.dilate(noWhiteNoise, kernel, iterations=3)
	    	keypoints = detector.detect(dilated)

		

		if keypoints is not None:
			for point in keypoints:
				# Get coordinates, move from center to corner
				x = int(point.pt[0])-14
				y = int(point.pt[1])-14

				if x < 0:
					x = 0
				if y < 0:
					y = 0
	 
				# Remove most of unrelated bright spots and crop to only number area
				ret,image = cv2.threshold(gray,20,255,cv2.THRESH_TOZERO)
				croppedAreaOfNumber = image[y:y+28,x:x+28]

				# Check if it is close to line
				x = int(point.pt[0])
				y = int(point.pt[1])
				distanceFromEndOfLineTop = math.sqrt(math.pow((x-x1),2)+math.pow((y-y1),2))
				distanceFromEndOfLineBottom = math.sqrt(math.pow((x-x2),2)+math.pow((y-y2),2))
				directDistanceToLine = math.fabs(length-(distanceFromEndOfLineTop+distanceFromEndOfLineBottom))

				# Check for direct distance and if number is within line, not close outter
				if directDistanceToLine < 0.5:
					if ((x>x1 and x<x2) and (y<y1 and y>y2)):
						predictedNum = trainedModel.predict(croppedAreaOfNumber.reshape(1,-1))
						sumInVideo = sumInVideo + predictedNum

	capturing.release()
	resultsString += "\n" + media + "\t" + str(int(sumInVideo))

# Write to file
writeToFile(resultsString)

