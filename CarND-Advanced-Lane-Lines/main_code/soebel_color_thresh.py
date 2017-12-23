
"""
this file has the combination of sobel and channel thresholding

@author: atpandey
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from undist_warp import *
from color_thresh import *
from soebel import *


#!!!!this file is not needed as all the functions are incorporated in soebel.py!!!!
    
def color_soebel_xy(img,soebel_kernel=3,mag_threshold=(25, 255),show=False):
    '''
    this function applies soebelx/y on a color thresholded image
    and AND's x/y 
    
    '''
    sobelx_grad=sobel_grad(img, orient='x', sobel_kernel=soebel_kernel)
    sobely_grad=sobel_grad(img, orient='y', sobel_kernel=soebel_kernel)
    sobelx=abs_sobel(sobelx_grad,mag_threshold=mag_threshold,show=show)
    sobely=abs_sobel(sobely_grad,mag_threshold=mag_threshold,show=show)
    sobel_x_y=img_and(sobelx, sobely,show=show)
    
    return sobel_x_y
    
def color_soebel_magdir(img,soebel_kernel=3,mag_threshold=(100, 200),dir_thresh=(0.7, 1.0),show=False):
    '''
    this function applies soebelmag/dir on a color thresholded image
    and AND's mag/dir 
    
    '''
    sobelx_grad=sobel_grad(img, orient='x', sobel_kernel=soebel_kernel)
    sobely_grad=sobel_grad(img, orient='y', sobel_kernel=soebel_kernel)
    sobelmags=magnitude_soebel(sobelx_grad, sobely_grad, mag_threshold=mag_threshold,show=show)
    sobeldirs=direction_soebel(sobelx_grad, sobely_grad,dir_thresh=dir_thresh,show=show)
    
    sobel_mag_dir=img_and(sobelmags, sobeldirs,show=show)
    
    return sobel_mag_dir

def color_sxy_smagdir_or(sxy,smagdir,soebel_kernel=3,abs_threhold=(25,255),mag_threshold=(100, 200),dir_thresh=(0.7, 1.0),show=False):
    '''
    this function or's sxy and smag/dir images
    '''
    sobel_x_y=color_soebel_xy(sxy,soebel_kernel=soebel_kernel,mag_threshold=abs_threhold,show=show)
    sobel_mag_dir=color_soebel_magdir(smagdir,soebel_kernel=soebel_kernel,mag_threshold=mag_threshold,dir_thresh=dir_thresh,show=show)
    sxy_or_mds=img_or(sobel_x_y,sobel_mag_dir,show=show)
    return sxy_or_mds
    
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
    
    img_ar=[imgname1,imgname2,imgname3,imgname4,imgname5,imgname6,imgname7,imgname8,imgname9,imgname10,imgname11]
    


################################
#    src = np.float32([(258,682),
#                       (575,464),
#                      (707,464), 
#                      (1049,682)])
#    dst = np.float32([(450,720),
#                       (450,0),
#                      (1280-450,0),                      
#                      (1280-450,720)])
    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
    new_top_left=np.array([corners[0,0],0])
    new_top_right=np.array([corners[3,0],0])
    offset=[150,0]
    
    img_size = (1280, 720)
    src = np.float32([corners[0],corners[1],corners[2],corners[3]])
    dst = np.float32([corners[0]+offset,new_top_left+offset,new_top_right-offset ,corners[3]-offset]) 
####################################
  
    
    pts=np.int32(src)
    pts = pts.reshape((-1,1,2))
    
    imagea=mpimg.imread(imgname4)
    undistorted=undistort(imagea,dist_pickle=dist_pickle)
#    polyimgorig=cv2.polylines(undistorted,[pts],True,(0,0,255),2)
    warp,M,Minv=perspective_trf(undistorted,src=src,dest=dst,show=False)
    #hls l and labb combined thresholded
    hls_l_lab_b=hls_l_lab_b_img(warp,scope=0)
    c_s_xy=color_soebel_xy(hls_l_lab_b)
    c_s_magdir=color_soebel_magdir(hls_l_lab_b)
    c_s_or=color_sxy_smagdir_or(c_s_xy,c_s_magdir)
    #visualize
    figc, axsc = plt.subplots(2,3)
    figc.subplots_adjust(hspace = 0.5, wspace=0.2)
    axsc = axsc.ravel()
    axsc[0].imshow(warp)
    axsc[0].set_title('warped')
    axsc[1].imshow(hls_l_lab_b,cmap='gray')
    axsc[1].set_title('hls_l_lab_b ')
    axsc[2].imshow(c_s_xy,cmap='gray')
    axsc[2].set_title('c_s_xy')
    axsc[3].imshow(c_s_magdir,cmap='gray')
    axsc[3].set_title('c_s_magdir')
    axsc[4].imshow(c_s_or,cmap='gray')
    axsc[4].set_title('c_s_or')

    