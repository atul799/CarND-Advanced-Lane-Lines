# -*- coding: utf-8 -*-
"""
Find radius of curvature for left and right lanes
also find position of vehicles w.r.t. lane centre

@author: atpandey
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from undist_warp import *
from color_thresh import *
from soebel import *
from sliding_window import *
from lane_finder_prev_fit import *

def calc_curv_rad_and_center_dist(img,left_fit,right_fit, leftx,lefty, rightx,righty):
    '''
    input use formula to find lane angle from fit poly's
    also finds distance of vehicle from lane centre
    
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension, 
    #xm_per_pix = 3.7/700 # meters per pixel in x dimension, 
    xm_per_pix = 3.7/600
    left_curverad, right_curverad, center_dist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    h = img.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)

    left_fit_cr=[0.001,0.001,0.1]
    right_fit_cr=[0.001,0.001,0.1]
    

    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        #curvature is in meters
    
    # Distance from center is image x midpoint - mean of left_fit and right_fit  
    if left_fit_cr is not None and left_fit_cr is not None:
#        car_position = img.shape[1]/2
        car_position = img.shape[1]/2
        left_fit_x_int = left_fit[0]*h**2 + left_fit[1]*h + left_fit[2]
        right_fit_x_int = right_fit[0]*h**2 + right_fit[1]*h + right_fit[2]
        lane_center_position = (right_fit_x_int + left_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * xm_per_pix
        #print("centre dist:",center_dist,left_fit_x_int,right_fit_x_int)
    return left_curverad, right_curverad, center_dist


def curv_draw(img, curverad, center_dist):
    new_img = np.copy(img)
    h = new_img.shape[0]
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'    
    cv2.putText(new_img, "Curvature: %.1f m." % curverad, (50, 70), cv2.FONT_HERSHEY_DUPLEX, 1.3, (200, 255, 255), 2)
    cv2.putText(new_img, "offset from Centre: %.3f m towards %s." %( abs(center_dist),direction), (50, 170), cv2.FONT_HERSHEY_DUPLEX, 1.3, (200, 255, 255), 2)
    
    
    return new_img



###test
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
#    src = np.float32([(258,682),
#                       (575,464),
#                      (707,464), 
#                      (1049,682)])
#    dst = np.float32([(450,720),
#                       (450,0),
#                      (830,0),                      
#                      (830,720)])
#    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
#    new_top_left=np.array([corners[0,0],0])
#    new_top_right=np.array([corners[3,0],0])
#    offset=[150,0]
#    
#    img_size = (1280, 720)
#    src = np.float32([corners[0],corners[1],corners[2],corners[3]])
#    dst = np.float32([corners[0]+offset,new_top_left+offset,new_top_right-offset ,corners[3]-offset]) 
####################################
  
    

    #single image based testing   
    imagea=mpimg.imread(imgname1)
    imageb=np.copy(imagea)
    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
    new_top_left=np.array([corners[0,0],0])
    new_top_right=np.array([corners[3,0],0])
    offset=[150,0]
    
    img_size = (imagea.shape[1], imageb.shape[0])
    src = np.float32([corners[0],corners[1],corners[2],corners[3]])
    dst = np.float32([corners[0]+offset,new_top_left+offset,new_top_right-offset ,corners[3]-offset])
    pts=np.int32(src)
    pts = pts.reshape((-1,1,2))
    polyimg=cv2.polylines(imageb,[pts],True,(255,0,0),5)
    img_t, Minv = im_pipe(imagea,dist_pickle=dist_pickle,sobel=False,show=False)
    left_fit, right_fit,leftx,lefty,rightx,righty, draw,histogram=sliding_window_polyfit(img_t,margin=80,minpix=40,nwindows=10)
    out_img=visualize_sliding_window(img_t,Minv,left_fit, right_fit, leftx,lefty, rightx,righty, draw,histogram,show=False)
    unwarped_img=visualize_sliding_window_image(imagea,img_t,Minv,left_fit, right_fit)
    left_curverad, right_curverad, center_dist=calc_curv_rad_and_center_dist(img_t,left_fit,right_fit, leftx,lefty, rightx,righty)
    lannotated_img=curv_draw(unwarped_img,(left_curverad+right_curverad)/2 , center_dist)
    print('Radius of curvature left:', left_curverad, 'm,', 'right:',right_curverad, 'm')
    print('Distance from lane center for example:', center_dist, 'm')
    
    f,a=plt.subplots()
#    a.plot(histogram)
#    plt.xlim(0, 1280)
    a.imshow(lannotated_img)
    
    