import cv2
import numpy as np
import math
import math
from sklearn import linear_model

# util functions
# Based on the paper "Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution"
def ssc(keypoints, descriptors, num_ret_points, tolerance, cols, rows):
    exp1 = rows + cols + 2 * num_ret_points
    exp2 = (
        4 * cols
        + 4 * num_ret_points
        + 4 * rows * num_ret_points
        + rows * rows
        + cols * cols
        - 2 * rows * cols
        + 4 * rows * cols * num_ret_points
    )
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
    sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

    high = (
        sol1 if (sol1 > sol2) else sol2
    )  # binary search range initialization with positive solution
    low = math.floor(math.sqrt(len(keypoints) / num_ret_points))

    prev_width = -1
    selected_keypoints = []
    selected_descriptors = []
    result_list = []
    result = []
    complete = False
    k = num_ret_points
    k_min = round(k - (k * tolerance))
    k_max = round(k + (k * tolerance))

    while not complete:
        width = low + (high - low) / 2
        if (
            width == prev_width or low > high
        ):  # needed to reassure the same radius is not repeated again
            result_list = result  # return the keypoints from the previous iteration
            break

        c = width / 2  # initializing Grid
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [
            [False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_rows + 1)
        ]
        result = []

        for i in range(len(keypoints)):
            row = int(
                math.floor(keypoints[i].pt[1] / c)
            )  # get position of the cell current point is located at
            col = int(math.floor(keypoints[i].pt[0] / c))
            if not covered_vec[row][col]:  # if the cell is not covered
                result.append(i)
                # get range which current radius is covering
                row_min = int(
                    (row - math.floor(width / c))
                    if ((row - math.floor(width / c)) >= 0)
                    else 0
                )
                row_max = int(
                    (row + math.floor(width / c))
                    if ((row + math.floor(width / c)) <= num_cell_rows)
                    else num_cell_rows
                )
                col_min = int(
                    (col - math.floor(width / c))
                    if ((col - math.floor(width / c)) >= 0)
                    else 0
                )
                col_max = int(
                    (col + math.floor(width / c))
                    if ((col + math.floor(width / c)) <= num_cell_cols)
                    else num_cell_cols
                )
                for row_to_cover in range(row_min, row_max + 1):
                    for col_to_cover in range(col_min, col_max + 1):
                        if not covered_vec[row_to_cover][col_to_cover]:
                            # cover cells within the square bounding box with width w
                            covered_vec[row_to_cover][col_to_cover] = True

        if k_min <= len(result) <= k_max:  # solution found
            result_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1  # update binary search range
        else:
            low = width + 1
        prev_width = width

    for i in range(len(result_list)):
        selected_keypoints.append(keypoints[result_list[i]])
        selected_descriptors.append(descriptors[result_list[i]])

    return selected_keypoints,selected_descriptors

# Finding epipolar point
def epipole_point(lines):
    m1 =  - (lines[0][0]/lines[0][1])
    m2 =  - (lines[1][0]/lines[1][1])
    b1 =  - (lines[0][2]/lines[0][1])
    b2 =  - (lines[1][2]/lines[1][1]) 
    if m1==m2 or b1==b2:
        m2 = - (lines[2][0]/lines[2][1])
        b2 = - (lines[2][2]/lines[2][1])  
    x1 = (b1-b2)/(m2-m1)
    y1 = m1*x1 + b1
    return (x1,y1)

# cartesian2polar coordinates
def cartesian2polar(pts,fov):
    r_dash = []
    theta = []
    for i in pts:
        r_dash.append(math.sqrt((i[0]-fov[0])**2 + (i[1]-fov[1])**2))
        theta.append(math.degrees(math.atan((i[1]-fov[1])/(i[0]-fov[0]))))
    return r_dash,theta

# Ransac for choosing inlier points
def ransac(X,y):
    X = np.array(X).reshape(-1,1)
    y= np.array(y)
    ransac = linear_model.RANSACRegressor(min_samples=5)
    ransac.fit(X,y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    return y[inlier_mask],np.mean(y[inlier_mask]),np.std(y[inlier_mask])


# New function is based on polar coordinate over opencv functions
def detect_moving_mask_new(image_prev,image_curr,image_mask1,image_mask2,image_track1,image_track2):

    image_prev = cv2.cvtColor(image_prev,cv2.COLOR_BGR2GRAY )
    image_prev = cv2.equalizeHist(image_prev)
    image_curr = cv2.cvtColor(image_curr,cv2.COLOR_BGR2GRAY )
    image_curr = cv2.equalizeHist(image_curr)

    ### Fundamental matrix calculation for epilines ###
    # key point extraction using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image_prev, None)
    kp2, des2 = sift.detectAndCompute(image_curr, None)
    kernel = np.ones((7,7),np.uint8)
    erosion1 = cv2.erode(image_mask1,kernel,iterations = 1)
    erosion2 = cv2.erode(image_mask2,kernel,iterations = 1)

    image_mask1[image_mask1<=160] =0
    image_mask2[image_mask2<=160] =0
    # removing keypoints which lies on the dynamic mask
    kp1_new = []
    des1_new = []
    for k in range(len(kp1)):
        pt1 = (math.ceil(kp1[k].pt[1]),math.ceil(kp1[k].pt[0]))
        if (erosion1[pt1] == 255 and erosion2[pt1] == 255):
            kp1_new.append(kp1[k])
            des1_new.append(des1[k])

    kp2_new = []
    des2_new = []
    for k in range(len(kp2)):
        pt1 = (math.ceil(kp2[k].pt[1]),math.ceil(kp2[k].pt[0]))
        if (erosion1[pt1] == 255 and erosion2[pt1] == 255):
            kp2_new.append(kp2[k])
            des2_new.append(des2[k])

    sel_kp1, sel_des1 = ssc(kp1_new, des1_new, 1000, 0.1, image_prev.shape[1], image_prev.shape[0])
    sel_kp2, sel_des2 = ssc(kp2_new, des2_new , 1000, 0.1, image_curr.shape[1], image_curr.shape[0])

    # trying to find the matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(np.asarray(sel_des1,np.float32),np.asarray(sel_des2,np.float32),k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])
        
    if len(good) > 10:
        matches = np.asarray(good)
    else:
        matches = np.asarray(matches)
            
    if len(matches[:,0]) >= 8: # verification for homography matrix8 point algorithm
        src = np.float32([ sel_kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ sel_kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    F, mask = cv2.findFundamentalMat(np.int32(src),np.int32(dst),cv2.FM_LMEDS)

    ## finding focus of expansion based on the staionary points
    new_src=np.int32(src)
    new_src = new_src.reshape(len(src),2)
    new_dst=np.int32(dst)
    new_dst = new_dst.reshape(len(dst),2)
    lines1 = cv2.computeCorrespondEpilines(new_dst, 2,F)
    lines1 = lines1.reshape(-1,3)
    lines2 = cv2.computeCorrespondEpilines(new_src, 1,F)
    lines2 = lines2.reshape(-1,3)

    fov_pts2 = epipole_point(lines1)
    fov_pts1 = epipole_point(lines2)

    ## optical flow tracking points ##
    srcgray = image_prev.copy()
    dstgray = image_curr.copy()
    kernel = np.ones((5,5),np.uint8)
    erosion1_gray = cv2.erode(image_mask1,kernel,iterations = 1)
    erosion2_gray = cv2.erode(image_mask2,kernel,iterations = 1)
    srcgray[image_mask1==255] = 255
    dstgray[image_mask2==255] = 255

    color = np.random.randint(0,255,(1000,3))

    feature_params = dict( maxCorners = 200,
                        qualityLevel = 0.1,
                        minDistance = 3,
                        blockSize = 7)
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 5,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p0_old = cv2.goodFeaturesToTrack(srcgray, mask = None, **feature_params)

    if p0_old is not None:
        p0 = []
        for k in range(len(p0_old)):
            pt1 = (math.floor(p0_old[k][0][1]),math.floor(p0_old[k][0][0]))
            if (erosion1_gray[pt1] == 255):
                continue
            elif (erosion2_gray[pt1] == 255):
                continue
            else:
                p0.append(p0_old[k])

        p0_mod = []
        for i in p0:
            # Removing outer bottom features observed in images. Might cause mismatch
            if (i[0][0]) > 0.05*image_prev.shape[1] and (i[0][0]) < 0.95*image_prev.shape[1] and (i[0][1]) < 0.8*image_prev.shape[0] :
                p0_mod.append(i)
        p0_mod = np.array(p0_mod)
       
        
        if len(p0_mod) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(srcgray, dstgray,p0_mod, None, **lk_params)
            # Select good points
            if p1 is not None:
                p1_mod = np.array(p1)

                good_new = []
                good_old = []
                track_ref = []
                for i in range(len(p1_mod)):
                    dist = ((((p1_mod[i][0][0] - p0_mod[i][0][0] )**2) + ((p1_mod[i][0][1]-p0_mod[i][0][1] )**2) )**0.5) 
                    if (p1_mod[i][0][1] < image_prev.shape[0] and p1_mod[i][0][0] < image_prev.shape[1] and p0_mod[i][0][1] < image_prev.shape[0] and p0_mod[i][0][0] < image_prev.shape[1]):           
                        if (err[i] < 40) and (st[i]==1) and abs(dist) < 100 and image_track1[math.floor(p0_mod[i][0][1]), math.floor(p0_mod[i][0][0])]==image_track2[math.floor(p1_mod[i][0][1]),math.floor( p1_mod[i][0][0])] :
                            good_new.append(p1_mod[i][0])
                            good_old.append(p0_mod[i][0])
                            track_ref.append(image_track1[math.floor(p0_mod[i][0][1]), math.floor(p0_mod[i][0][0])])
            
            invalid_points = []
            for i in np.unique(image_track1):
                if i == 0:
                    track_ref = np.array(track_ref)
                    idx = np.where(track_ref == i)[0]
                    for j in idx:
                        invalid_points.append(j)
                else:
                    track_ref = np.array(track_ref)
                    idx = np.where(track_ref == i)[0]
                    positive_list = []
                    negative_list = []
                    for j in idx:
                        if math.degrees(math.atan((good_new[j][1]-good_old[j][1])/(good_new[j][0]-good_old[j][0]))) > 0:
                            positive_list.append(j)
                        else:
                            negative_list.append(j)
                    if len(positive_list)>len(negative_list):
                        for k in negative_list:
                            invalid_points.append(k)
                    else:
                        for k in positive_list:
                            invalid_points.append(k)
                    
            final_old = []
            final_new = []
            for i in range(len(good_old)):
                if i not in invalid_points:
                    final_old.append(good_old[i])
                    final_new.append(good_new[i])

            if len(final_new)>6:
                ### epipolar constraint ##
                pts1 = np.int32(final_old)
                pts2 = np.int32(final_new)

                extr_points = []
                # checking car movement based on Focus of expansion
                if (fov_pts2[0] < image_curr.shape[1] and fov_pts2[0] > 0 and fov_pts2[1] < image_curr.shape[0] and fov_pts2[1] > 0):
                    # car moves straight
                    if np.linalg.norm(np.array(fov_pts2)-np.array(fov_pts1)) < 3:
                        r_dash1,theta1 = cartesian2polar(pts1,fov_pts2)
                        r_dash2,theta2 = cartesian2polar(pts2,fov_pts2)
                        for i in range(len(pts2)):
                            if abs(r_dash1[i]-r_dash2[i] )>2 and abs((theta1[i]-theta2[i])) > 3:
                                extr_points.append(pts2[i])
                    # car moves in curve
                    else:
                        r_dash1,theta1 = cartesian2polar(pts1,fov_pts2)
                        r_dash2,theta2 = cartesian2polar(pts2,fov_pts1)
                        for i in range(len(pts2)):
                            if abs(r_dash1[i]-r_dash2[i] )>5 and abs((theta1[i]-theta2[i])) > 3:
                                extr_points.append(pts2[i])

                # car stopped
                else:
                    for  i in range(len(pts2)):
                        if abs(math.sqrt( (pts2[i][0] - pts1[i][0])**2 + (pts2[i][1] - pts1[i][1])**2 )) >5:
                            extr_points.append(pts2[i])

            else:
                extr_points=[]

        else:
            extr_points=[]
        
    else:
            extr_points=[]
        
    return extr_points   
