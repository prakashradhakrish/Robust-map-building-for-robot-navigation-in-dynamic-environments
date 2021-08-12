import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

def detect_moving_mask(image_prev,image_curr,image_mask1,image_mask2):

	image_prev = cv2.cvtColor(image_prev,cv2.COLOR_BGR2GRAY )
	image_prev = cv2.equalizeHist(image_prev)
	image_curr = cv2.cvtColor(image_curr,cv2.COLOR_BGR2GRAY )
	image_curr = cv2.equalizeHist(image_curr)

	### Fundamental matrxi calculation for epilines ###
	# key point extraction using SIFT
	sift = cv2.SIFT_create()
	kp1, des1 = sift.detectAndCompute(image_prev, None)
	kp2, des2 = sift.detectAndCompute(image_curr, None)
	image_mask2[image_mask2<=60] =255
	kernel = np.ones((7,7),np.uint8)
	erosion1 = cv2.erode(image_mask1,kernel,iterations = 1)
	erosion2 = cv2.erode(image_mask2,kernel,iterations = 1)


	# removing kepypoints which lies on the dynamic mask
	kp1_new = []
	des1_new = []
	for k in range(len(kp1)):
		pt1 = (math.ceil(kp1[k].pt[1]),math.ceil(kp1[k].pt[0]),0)
		if (erosion1[pt1] == 155 ) or (erosion2[pt1] == 155) or (erosion1[pt1] == 55) or (erosion2[pt1] == 55):
			continue
		else:
			kp1_new.append(kp1[k])
			des1_new.append(des1[k])

	kp2_new = []
	des2_new = []
	for k in range(len(kp2)):
		pt1 = (math.ceil(kp2[k].pt[1]),math.ceil(kp2[k].pt[0]),0)
		if (erosion1[pt1] == 155 ) or (erosion2[pt1] == 155) or (erosion1[pt1] == 55) or (erosion2[pt1] == 55):
			continue
		else:
			kp2_new.append(kp2[k])
			des2_new.append(des2[k])

	# trying to find the matches
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(np.asarray(des1_new,np.float32),np.asarray(des2_new,np.float32),k=2)
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])
		
	if len(good) > 10:
	    matches = np.asarray(good)
	else:
	    matches = np.asarray(matches)
	        
	if len(matches[:,0]) >= 8: # verification for homography matrix8 point algorithm
	    src = np.float32([ kp1_new[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
	    dst = np.float32([ kp2_new[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

	#print("Actual_matches: ",len(src),len(dst))
	    
	F, mask = cv2.findFundamentalMat(np.int32(src),np.int32(dst),cv2.FM_LMEDS)
	
	## optical flow tracking points ##
	srcgray = image_prev.copy()
	dstgray = image_curr.copy()
	image_mask1_gray = cv2.cvtColor(image_mask1,cv2.COLOR_BGR2GRAY )
	image_mask2_gray = cv2.cvtColor(image_mask2,cv2.COLOR_BGR2GRAY )
	kernel = np.ones((5,5),np.uint8)
	erosion1_gray = cv2.erode(image_mask1_gray,kernel,iterations = 1)
	erosion2_gray = cv2.erode(image_mask2_gray,kernel,iterations = 1)
	srcgray[erosion1_gray>155] = 255
	dstgray[erosion2_gray>155] = 255


	color = np.random.randint(0,255,(1000,3))

	feature_params = dict( maxCorners = 200,
		               qualityLevel = 0.01,
		               minDistance = 3,
		               blockSize = 7)
	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (7,7),
		          maxLevel = 5,
		          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	
	p0_old = cv2.goodFeaturesToTrack(srcgray, mask = None, **feature_params)

	if p0_old is not None:
		p0 = []
		for k in range(len(p0_old)):
			pt1 = (math.ceil(p0_old[k][0][1]),math.ceil(p0_old[k][0][0]))
			if (erosion1_gray[pt1] == 255):
				continue
			elif (erosion2_gray[pt1] == 255):
				continue
			elif ((erosion1_gray[pt1] == 55) and (erosion2_gray[pt1] == 155)) or ((erosion1_gray[pt1] == 155) and (erosion2_gray[pt1] == 155)):
				p0.append(p0_old[k])

		p0=np.array(p0)
		mask = np.zeros_like(image_prev)
		#print("P0 count: ", len(p0))
		
		if len(p0) > 0:
			p1, st, err = cv2.calcOpticalFlowPyrLK(srcgray, dstgray,p0, None, **lk_params)
			# Select good points
			if p1 is not None:
				good_new = []
				good_old = []
				for i in range(len(p1)):
					if (err[i] < 60) and (st[i]==1):
						good_new.append(p1[i][0])
						good_old.append(p0[i][0])
			# draw the tracks
			for i,(new,old) in enumerate(zip(good_new, good_old)):
				a,b = new.ravel()
				c,d = old.ravel()
				mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
				mask = cv2.circle(mask,(int(c),int(d)),5,color[i].tolist(),-1)
				
			if len(good_new)>0:
				### epipolar constraint ##
				pts1 = np.int32(good_old)
				pts2 = np.int32(good_new)

				# Find epilines corresponding to points in right image (second image) and
				# drawing its lines on left image
				lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
				lines1 = lines1.reshape(-1,3)
				#img5,img6 = drawlines(image_prev,image_curr,lines1,pts1,pts2)
				# Find epilines corresponding to points in left image (first image) and
				# drawing its lines on right image
				lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
				lines2 = lines2.reshape(-1,3)

				extr_points = []
				counter = 0
				for l1, p1_dash in zip(lines2,pts2):
					distance = (l1[0]*p1_dash[0])+(l1[1]*p1_dash[1])+l1[2]
					if abs(distance) > 5:		
						extr_points.append(p1_dash)
					counter = counter +1
			else:
				extr_points=[]
			#print("Actual points: ", len(extr_points))
		else:
			extr_points=[]
		
		#image_actual = cv2.cvtColor(image_curr,cv2.COLOR_GRAY2BGR)
		#color = tuple(np.random.randint(0,255,3).tolist())
		#for i in extr_points:
			#print(i)
			#image_actual = cv2.circle(image_actual,tuple(i),5,color,-1)
		#cv2.imwrite("image_actual.png",image_actual)
	else:
		extr_points=[]
	return extr_points




















