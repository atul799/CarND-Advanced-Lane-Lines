
"""
complete pipeline
im warp
color and sobel threshold
sliding window
nonsliding
radius and curvature measurement
lineclass

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
from rad_curve import *
from line_class import *


#define Line class objects fro left and right lane
left_line=Line()
right_line = Line()



def pipeline(img):
    '''
    this functions is the pipeline for processing image or videos to find lane lines and annotate them on original image
    '''
    
    #load calibration data
    calib_file='../camera_cal/calib_pickle.p'
    dist_pickle = pickle.load( open(calib_file , "rb" ) ) 
    new_img = np.copy(img)
    
    #apply im_pipe from soebel.py file
    #im_pipe has undistrtion,warp perspective,color channel and soebel thresholding steps
    img_bin, Minv = im_pipe(new_img,dist_pickle=dist_pickle,sobel=False,show=False)
    
    # if  lane lines found in prev frame, use lanefinder_prev_fit, else use sliding_window_polyfit window
    if not left_line.detected or not right_line.detected:
        left_fit, right_fit,leftx,lefty,rightx,righty, draw,histogram=sliding_window_polyfit(img_bin,margin=120,minpix=60,nwindows=9)
        
    else:
        left_fit, right_fit, leftx, lefty,rightx,righty=lanefinder_prev_fit(img_bin,left_line.best_fit, right_line.best_fit, margin=120)
        
    #update line classes
    left_line.update_fit(left_fit)
    right_line.update_fit(right_fit)
    
    # draw the current best fit if it exists
    if left_line.best_fit is not None and right_line.best_fit is not None:
        unwarped_img=visualize_sliding_window_image(img,img_bin,Minv,left_line.best_fit, right_line.best_fit)
        left_curverad, right_curverad, center_dist=calc_curv_rad_and_center_dist(img_bin,left_line.best_fit,right_line.best_fit, leftx,lefty, rightx,righty)
        #print("left_curverad, right_curverad, center_dist",left_curverad, right_curverad, center_dist)
        
        img_out=curv_draw(unwarped_img,(left_curverad+right_curverad)/2 , center_dist)
    else:
        #print("else")
        img_out = new_img
        
    return img_out

##test function
#####################################
if __name__=='__main__':
       

    
    left_line = Line()
    right_line = Line()
    vid1 = '../project_video.mp4'
    voutput1='./project_video_annotated.mp4' 
    if os.path.isfile(voutput1):
        os.remove(voutput1) 
    video_clip = VideoFileClip(vid1).subclip(0,2)
    processed_video = video_clip.fl_image(pipeline)
    processed_video.write_videofile(voutput1, audio=False)    

#    vid2 = '../challenge_video.mp4'
#    voutput2='./challenge_video_annotated.mp4' 
#    if os.path.isfile(voutput2):
#        os.remove(voutput2) 
#    video_clip = VideoFileClip(vid2)#.subclip(0,2)
#    processed_video = video_clip.fl_image(pipeline)
#    processed_video.write_videofile(voutput2, audio=False)    
##
#    vid3 = '../harder_challenge_video.mp4'
#    voutput3='./harder_challenge_video_annotated.mp4' 
#    scope=500
#    if os.path.isfile(voutput3):
#        os.remove(voutput3) 
#    video_clip = VideoFileClip(vid3).subclip(25,42)
#    processed_video = video_clip.fl_image(pipeline)
#    processed_video.write_videofile(voutput3, audio=False)  