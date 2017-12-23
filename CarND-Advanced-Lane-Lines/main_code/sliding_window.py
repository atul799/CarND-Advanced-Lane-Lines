# -*- coding: utf-8 -*-
"""
sliding window implementation for finding initial lanes lines or 
if lanes are not identified in more than certain number of images

@author: atpandey
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from undist_warp import *
from color_thresh import *
from soebel import *

def sliding_window_polyfit(img,margin=100,minpix=50,nwindows=9):
    '''
    this function uses histogram to find lane lines using a sliding window approach
    then generates a polynomial fir for lanes lines
    inputs are:   image, margin and minpix (minimum number of pix to cosider)
    outputa are: fit values,coordinates for rectangle to draw around lanes and histogram
    '''
    # gen a histogram with bottom half image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Find  peaks in the histograms
    #mid point is the centre of x
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    # Choose the number of sliding windows
    nwindows = nwindows
    
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = margin
    # Set minimum number of pixels found to recenter window
    minpix = minpix
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # place holder for image visualization
    draw = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        draw.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    
    return left_fit, right_fit,leftx,lefty,rightx,righty, draw,histogram


def visualize_sliding_window(img,Minv,left_fit, right_fit, leftx,lefty, rightx,righty, draw,histogram,show=False):
    '''
    this function takes fit and x,y pos from sliding window finder function above and draws them on the 
    warped image
    '''
    #height and width of image used for generating init points
    h,w = img.shape[:2]
    #use fit data to generate init points x and y
    left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
    right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]

    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((img, img, img))*255)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    for rect in draw:
    # Draw the windows on the visualization image
        cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2) 
        cv2.rectangle(out_img,(rect[4],rect[0]),(rect[5],rect[1]),(0,255,0), 2) 
    # Identify the x and y positions of all nonzero pixels in the image

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    if show:
        ff,axs=plt.subplots(1,2)
        axs[0].imshow(out_img)
        axs[0].plot(left_fitx, ploty, color='yellow')
        axs[0].plot(right_fitx, ploty, color='yellow')
        axs[1].imshow(img)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return out_img

def visualize_sliding_window_image(img,warp_bin,Minv,left_fit, right_fit):
    '''
    this function draws polygon between lanes on orig image
    '''
    #if lanes not found ret orig image
    new_img = np.copy(img)
    if left_fit is None or right_fit is None:
        return img
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warp_bin).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Color in left and right line pixels
    (h,w) = (img.shape[0],img.shape[1])
    # Generate x and y values for plotting
    ploty = np.linspace(0, h-1, num=h)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # #generate points and format them to appy on fillpoly
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h)) 
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    return result


#test functions
if __name__=='__main__':
    
    calib_file='../camera_cal/calib_pickle.p'
    dist_pickle = pickle.load( open(calib_file , "rb" ) ) 
    
    #straight lines1
    imgname1='../test_images/straight_lines1.jpg'
    #straight lines2
    imgname2='../test_images/straight_lines2.jpg'
    #lane bending left
    imgname3='../test_images/test1.jpg'
    #lane bending left
    imgname4='../test_images/test2.jpg'
    #lane bending right
    imgname5='../test_images/test3.jpg'    
    #curve and shadow
    imgname6='../test_images/test4.jpg'
    #curve and shadow
    imgname7='../test_images/test5.jpg'
    #curve and shadow
    imgname8='../test_images/test6.jpg'    
    #challengevideo @3s
    imgname9='../main_code/challenge_video_3s.jpg'
    #harder challenge @3s
    imgname10='../main_code/harder_challenge_video_3s.jpg'
    #project vid @30s
    imgname11='../main_code/project_video_30s.jpg'
    
    #image = mpimg.imread(imgname2)
    
    #img_ar=[imgname1,imgname2,imgname3,imgname4,imgname5,imgname6,imgname7,imgname8,imgname9,imgname10,imgname11]
    img_ar=[imgname1,imgname4,imgname5,imgname7,imgname9,imgname10,imgname11]


################################
    src = np.float32([(258,682),
                       (575,464),
                      (707,464), 
                      (1049,682)])
    dst = np.float32([(450,720),
                       (450,0),
                      (830,0),                      
                      (830,720)])
####################################
  
    
    pts=np.int32(src)
    pts = pts.reshape((-1,1,2))
#    #single image based testing   
#    imagea=mpimg.imread(imgname11)
#    imageb=np.copy(imagea)
#    polyimg=cv2.polylines(imageb,[pts],True,(255,0,0),5)
#    img_t, Minv = im_pipe(imagea,dist_pickle=dist_pickle,sobel=False,show=False)
#    left_fit, right_fit,leftx,lefty,rightx,righty, draw,histogram=sliding_window_polyfit(img_t,margin=80,minpix=40,nwindows=9)
#    out_img=visualize_sliding_window(img_t,Minv,left_fit, right_fit, leftx,lefty, rightx,righty, draw,histogram,show=True)
#    f,a=plt.subplots()
#    a.plot(histogram)
#    plt.xlim(0, 1280)
##############################
    # Set up plot
    fig, axs = plt.subplots(len(img_ar),4, figsize=(25, 30))
    fig.subplots_adjust(hspace = .01, wspace=.1)
    axs = axs.ravel()
    plt.tight_layout()
                  
    i = 0
    for image in img_ar:
        img = mpimg.imread(image)
        imga=np.copy(img)
        polyimg=cv2.polylines(imga,[pts],True,(255,0,0),5)
        img_t, Minv = im_pipe(img,dist_pickle=dist_pickle,sobel=False,show=False)
        left_fit, right_fit,leftx,lefty,rightx,righty, draw,histogram=sliding_window_polyfit(img_t,margin=80,minpix=40,nwindows=9)
        out_img=visualize_sliding_window(img_t,Minv,left_fit, right_fit, leftx,lefty, rightx,righty, draw,histogram)
        unwarped_img=visualize_sliding_window_image(img,img_t,Minv,left_fit, right_fit)
#        axs[i].imshow(img)
#        axs[i].axis('off')
#        i += 1
        axs[i].imshow(polyimg)
        axs[i].axis('off')
        i += 1
        axs[i].imshow(img_t, cmap='gray')
        axs[i].axis('off')
        i += 1    
        axs[i].imshow(out_img, cmap='gray')
        axs[i].axis('off')
        i += 1
        axs[i].imshow(unwarped_img)
        #plt.xlim(0, 1280)
        i += 1
