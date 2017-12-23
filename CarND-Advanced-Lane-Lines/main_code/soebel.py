# -*- coding: utf-8 -*-
"""
This module  experiments with different color spaces and applying soebels on them
@author: atpandey
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from undist_warp import *
from color_thresh import *

####Soebel gradient and thresholding related functions
def sobel_grad(gray, orient='x', sobel_kernel=3,show=False):
    
    '''
    this function creates  soebelx and y image and applies absolute thresholds
    image : input image (as grayscaled image)
    soebel_kernel: kernel size
    orient: 'x' soebel gradient in x dir , 'y' soebel gradient in y dir
    mag_threshold: tuple of thresholds
    
    returns:
        binary_output: image with gradient and threshold applied
    
    '''
    
    #expected gray scaled image, if image has >1 channel issue error and return 1
    if len(gray.shape) != 2 :
        print("Err:Expected gray channel image in sobel_grad function")
        return -99
    
    
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    if orient=='y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    
    if show:
        fs,axs=plt.subplots()
        axs.imshow(sobel,cmap='gray')              
        axs.set_title("Soebel"+orient)          

    
    return sobel

def abs_sobel(sobel,mag_threshold=(0, 255),show=False):
    '''
    this function applies threshold on absolute values of sobel gradient
    sobel: inpuit sobel image, single channel
    mag_threshold : tuple pair to apply threshold on abs gradient
    returns:
        sobel : gradient treshold applied on sobel
    '''
    if len(sobel.shape) != 2 :
        print("Err:Expected single channel image in abs_soebel function")
        return -99
    abs_sobel = np.absolute(sobel)

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel,dtype=np.uint8)

    binary_output[(scaled_sobel >= mag_threshold[0]) & (scaled_sobel <= mag_threshold[1])] = 1
    
    if show:
        fabs,axabs=plt.subplots()
        axabs.imshow(binary_output)              
        axabs.set_title("Soebel")          
    return binary_output


def magnitude_soebel(sobelx, sobely, mag_threshold=(0, 255),show=False):
    '''
    soebelx:input soebelx gradient thresholded img
    soebely:input soebely gradient thresholded img
    
    mag_threshold: tuple of thresholds
    
    returns:
        binary_output: image with magnitude threshold applied
    
    '''
    #expected single channel soebel gradient with threshold in x and y applied,
    #if image has >1 channel issue error and return 1
    if (len(sobelx.shape) !=2 | len(sobely.shape) !=2):
        print("Err:Expected single channel image in magnitude_soebel function")
        return -99
    
    
    abs_sobelx = np.square(sobelx)
    abs_sobely = np.square(sobely)
    abs_sobel=np.sqrt(abs_sobelx+abs_sobely)
    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    binary_output = np.zeros_like(scaled_sobel,dtype=np.uint8)
    binary_output[(scaled_sobel >= mag_threshold[0]) & (scaled_sobel <= mag_threshold[1])] = 1
    
                  
    if show:
        fabs,axmags=plt.subplots()
        axmags.imshow(binary_output)              
        axmags.set_title("Soebel mag") 
    
    return binary_output



def direction_soebel(sobelx, sobely,dir_thresh=(0, np.pi/2),show=False):
    '''
    this function creates  soebelx and y image and applies direction based thresholds
    gray : input image (as grayscaled image)
    soebel_kernel: kernel size
    dir_threshold: tuple of thresholds in radians
    
    returns:
        binary_output: image with gradient and threshold applied
    
    '''
    #expected single channel soebel gradient with threshold in x and y applied,
    #if image has >1 channel issue error and return 1
    if (len(sobelx.shape) !=2 | len(sobely.shape) !=2):
        print("Err:Expected single channel image in magnitude_soebel function")
        return -99
    
    abs_sobelx=np.abs(sobelx)
    abs_sobely=np.abs(sobely)
    dir_gradient=np.arctan2(abs_sobely,abs_sobelx)
    
    binary_output = np.zeros_like(dir_gradient,dtype=np.uint8)
    
    binary_output[(dir_gradient >= dir_thresh[0]) & (dir_gradient <= dir_thresh[1])] = 1
    
    
    if show:
        fdirs,axdirs=plt.subplots()
        axdirs.imshow(binary_output,cmap='gray')              
        axdirs.set_title("Soebel dir") 
    
    return binary_output



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
    sobelmags=magnitude_soebel(sobelx, sobely, mag_threshold=mag_threshold,show=show)
    sobeldirs=direction_soebel(sobelx, sobely,dir_thresh=dir_thresh,show=show)
    
    sobel_mag_dir=img_and(sobelmags, sobeldirs,show=show)
    
    return sobel_mag_dir

def color_sxy_smagdir_or(image,soebel_kernel=3,abs_threhold=(25,255),mag_threshold=(100, 200),dir_thresh=(0.7, 1.0),show=False):
    '''
    this function or's sxy and smag/dir images
    '''
    sobel_x_y=color_soebel_xy(image,soebel_kernel=soebel_kernel,mag_threshold=abs_threhold,show=show)
    sobel_mag_dir=color_soebel_magdir(image,soebel_kernel=soebel_kernel,mag_threshold=mag_threshold,dir_thresh=dir_thresh,show=show)
    sxy_or_mds=img_or(sobel_x_y,sobel_mag_dir,show=show)
    return sxy_or_mds    


def im_pipe(image,dist_pickle,sobel=False,show=False):
    '''
    this function build color and sobele threshold pipeline
    image: image data
    src: 4 src points for warping
    dst: 4 dest points for warping
    dist_pickle: calibration data
    sobel: True apply soebel/False don't
    returns :
        c_s_xy_magdir: color+soebel applied image if sobel==True else hls_l_lab_b
        Minv: invesre perspective matrix
    
    '''
    #undistort image
    undistorted=undistort(image,dist_pickle=dist_pickle)
    #draw apolygon for warping
#    polyimgorig=cv2.polylines(undistorted,[pts],True,(0,0,255),2)
    #warp pespective transform
    #warp,M,Minv=perspective_trf(undistorted,src=src,dest=dst,show=show)
    warp,M,Minv=perspective_trf(undistorted,show=show)
    #hls l and labb combined thresholded
    hls_l_lab_b=hls_l_lab_b_img(warp,scope=0)
    if sobel:
        c_s_xy_magdir=color_sxy_smagdir_or(hls_l_lab_b,show=show)
        return c_s_xy_magdir,Minv
    else:
        return hls_l_lab_b,Minv
#################






#test the code
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
                      (1280-450,0),                      
                      (1280-450,720)])
####################################
  
    
    pts=np.int32(src)
    pts = pts.reshape((-1,1,2))
#single image based testing   
    imagea=mpimg.imread(imgname4)
    undistorted=undistort(imagea,dist_pickle=dist_pickle)
#    polyimgorig=cv2.polylines(undistorted,[pts],True,(0,0,255),2)
    warp,M,Minv=perspective_trf(undistorted,show=False)
    #hls l and labb combined thresholded
    hls_l_lab_b=hls_l_lab_b_img(warp,scope=0)
    sobelx_grad=sobel_grad(hls_l_lab_b, orient='x', sobel_kernel=3)
    sobely_grad=sobel_grad(hls_l_lab_b, orient='y', sobel_kernel=3)
    sobelx=abs_sobel(sobelx_grad,mag_threshold=(25, 255),show=False)
    sobely=abs_sobel(sobely_grad,mag_threshold=(25, 255),show=False)
    sobelmags=magnitude_soebel(sobelx, sobely, mag_threshold=(100, 200),show=False)
    sobeldirs=direction_soebel(sobelx, sobely,dir_thresh=(0.7, 1.0),show=False)
    
    sobel_x_y=color_soebel_xy(hls_l_lab_b)
    sobel_mag_dir=color_soebel_magdir(hls_l_lab_b)
    sxy_or_mds=color_sxy_smagdir_or(hls_l_lab_b)

    #visualize
    figc, axsc = plt.subplots(3,3)
    figc.subplots_adjust(hspace = 0.5, wspace=0.2)
    axsc = axsc.ravel()
    axsc[0].imshow(warp)
    axsc[0].set_title('warped')
    axsc[1].imshow(hls_l_lab_b,cmap='gray')
    axsc[1].set_title('hls_l_lab_b ')
    axsc[2].imshow(sobelx,cmap='gray')
    axsc[2].set_title('abssobelx')
    axsc[3].imshow(sobely,cmap='gray')
    axsc[3].set_title('abssobely')
    axsc[4].imshow(sobelmags,cmap='gray')
    axsc[4].set_title('sobelmags')
    axsc[5].imshow(sobeldirs,cmap='gray')
    axsc[5].set_title('sobeldirs ')
    axsc[6].imshow(sobel_x_y,cmap='gray')
    axsc[6].set_title('sobel_x_y')
    axsc[7].imshow(sobel_mag_dir,cmap='gray')
    axsc[7].set_title('sobel_mag_dir')  
    axsc[8].imshow(sxy_or_mds,cmap='gray')
    axsc[8].set_title('sxy_or_mds')
    
    #channel thresholding is good enough
    #soebel doesn't help, it reduces pixel density!!
    #and of mag and dir produces black image!!
###########################################################
#   #image pipe testing
#    c_soebel,Minv1=im_pipe(imagea,src=src,dst=dst,dist_pickle=dist_pickle,sobel=True)
#    c_only,Minv2=im_pipe(imagea,src=src,dst=dst,dist_pickle=dist_pickle,sobel=False)
#    #visualize
#    figc, axsc = plt.subplots(1,3)
#    figc.subplots_adjust(hspace = 0.5, wspace=0.2)
#    #axsc = axsc.ravel()
#    axsc[0].imshow(imagea)
#    axsc[0].set_title('img')
#    axsc[1].imshow(c_soebel,cmap='gray')
#    axsc[1].set_title('c_soebel ')
#    axsc[2].imshow(c_only,cmap='gray')
#    axsc[2].set_title('c_only')

#########################################################
                                          
    # Set up plot
#    fig, axs = plt.subplots(len(img_ar),2, figsize=(25, 30))
#    fig.subplots_adjust(hspace = .01, wspace=.1)
#    axs = axs.ravel()
#    plt.tight_layout()
#                  
#    i = 0
#    for image in img_ar:
#        img = mpimg.imread(image)
#        img_t, Minv = im_pipe(img,src=src,dst=dst,dist_pickle=dist_pickle,sobel=False,show=False)
#        polyimg=cv2.polylines(img,[pts],True,(255,0,0),5)
##        axs[i].imshow(img)
##        axs[i].axis('off')
##        i += 1
#        axs[i].imshow(polyimg)
#        axs[i].axis('off')
#        i += 1
#        axs[i].imshow(img_t, cmap='gray')
#        axs[i].axis('off')
#        i += 1    