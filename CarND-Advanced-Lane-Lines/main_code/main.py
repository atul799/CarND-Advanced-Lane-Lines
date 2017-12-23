
"""
This file uses all the different ssteps required for Lane finding including
camera calibration
Steps are:
    1.camera calibration
    2. undistort image
    3. warp image
    4. apply color and sobel thresholding
    5. sliding window or use previous fit to find lane lines
    6. measure curvature 
    7. annotate image/frame with polygon on found lanes and curvature

@author: atpandey
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import glob
import pickle
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

###import packages for this project
from cam_calibration import *
from undist_warp import *
from color_thresh import *
from soebel import *
from sliding_window import *
from lane_finder_prev_fit import *
from rad_curve import *
from line_class import *
from pipeline import *


###calibrate camera################
images = glob.glob('../camera_cal/*.jpg')
    

#horizontal corners in chess board
x_corners=9
#vertical corners on chessboard
y_corners=6
#disabled after first calibration
#objpoints,imgpoints=chessboard_corners(images,x_corners,y_corners,show=False)

#########################
#undistort --> warp+soebel --> lane finder is in function 
#pipeline in file pipeline.py


#Apply pipeline on the videos in the project

#use project video and annotate
vid1 = '../project_video.mp4'
voutput1='./project_video_annotated.mp4' 
if os.path.isfile(voutput1):
    os.remove(voutput1)
#read video 
video_clip = VideoFileClip(vid1)
#process video clip and return from function pipeline
processed_video = video_clip.fl_image(pipeline)
#write video stream to file
processed_video.write_videofile(voutput1, audio=False)    

#challenge video
vid2 = '../challenge_video.mp4'
voutput2='./challenge_video_annotated.mp4' 
if os.path.isfile(voutput2):
    os.remove(voutput2) 
video_clip = VideoFileClip(vid2)#.subclip(0,2)
processed_video = video_clip.fl_image(pipeline)
processed_video.write_videofile(voutput2, audio=False)    
#harder challenge video
vid3 = '../harder_challenge_video.mp4'
voutput3='./harder_challenge_video_annotated.mp4' 
scope=500
if os.path.isfile(voutput3):
    os.remove(voutput3) 
video_clip = VideoFileClip(vid3)#.subclip(25,42)
processed_video = video_clip.fl_image(pipeline)
processed_video.write_videofile(voutput3, audio=False)  


#%%
from moviepy.editor import *
clip = (VideoFileClip('./project_video_annotated.mp4')
        .subclip((1),(5))
        .resize(0.3))
clip.write_gif('proj_vid_ann.gif')