import cv2

def getMyBlobParams():
	params = cv2.SimpleBlobDetector_Params()
	params.blobColor = 255
	params.filterByColor = True
	params.filterByArea = False
	params.minArea = 180
	params.maxArea = 500
	params.filterByCircularity = False
	params.filterByConvexity = False
	params.filterByInertia = False
	return params
