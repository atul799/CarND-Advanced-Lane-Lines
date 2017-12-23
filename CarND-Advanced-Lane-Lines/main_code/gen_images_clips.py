
"""
this code piece generates smaller clips as well as images from challenge and harder
challenge video
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




if __name__=='__main__':
    video1 = '../project_video.mp4'
    video2 = '../challenge_video.mp4'
    video3 = '../harder_challenge_video.mp4'
    voutput1 = 'clip_7_11s.mp4'
    voutput2 = 'clip_22_32s.mp4'
    voutput3 = 'clip_35_47s.mp4'
    voutput4 = 'challenge_clip_0_7s.mp4'
#    #    video = '../challenge_video.mp4'
#    #    voutput = './cvid_annotated.mp4'    
#    #video = '../harder_challenge_video.mp4'
#    #voutput = './hcvid_annotated.mp4'
#    if os.path.isfile(voutput1):
#        os.remove(voutput1)
#    if os.path.isfile(voutput2):
#        os.remove(voutput2)   
#    if os.path.isfile(voutput3):
#        os.remove(voutput3) 
#    if os.path.isfile(voutput4):
#        os.remove(voutput4)      
##    video_clip = VideoFileClip(video)
##    processed_clip = video_clip.fl_image(frame_processor)
##    %time    processed_clip.write_videofile(voutput, audio=False)
#    clip1 = VideoFileClip(video1).subclip(7,11)
#    clip1.write_videofile(voutput1, audio=False)
#    clip2 = VideoFileClip(video1).subclip(22,32)
#    clip2.write_videofile(voutput2, audio=False)    
#    clip3 = VideoFileClip(video1).subclip(35,47)
#    clip3.write_videofile(voutput3, audio=False) 
#    
#    clip4 = VideoFileClip(video2).subclip(0,7)
#    clip4.write_videofile(voutput4, audio=False)
############
#lets' generate some images to generalize image thresholding
vid1=video1
video_clip1= VideoFileClip(vid1)
video_clip1.save_frame('./project_video_30s.jpg', t='00:00:30')

vid2 = video2
video_clip2= VideoFileClip(vid2)
video_clip2.save_frame('./challenge_video_3s.jpg', t='00:00:03')

vid3 = video3
video_clip3= VideoFileClip(vid3)
video_clip3.save_frame('./harder_challenge_video_3s.jpg', t='00:00:03')