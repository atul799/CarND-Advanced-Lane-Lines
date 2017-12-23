
"""
this module is for  experimentation with color channels

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


#import undistort and warpperspective function code
from undist_warp import *

def color_space_converter(image,space='hsv'):
    '''
    this function expects a 3 channel image in rgb format
    and converts it to specified format
    image: input rgb image
    space" color space spec, allowed spaces are, bgr,hsv,hls and Lab,gray,yuv
    '''
    allowed_color_space={'bgr':cv2.COLOR_RGB2BGR,'hsv':cv2.COLOR_RGB2HSV,'hls':cv2.COLOR_RGB2HLS,'Lab':cv2.COLOR_RGB2Lab, \
    'gray':cv2.COLOR_RGB2GRAY,'yuv':cv2.COLOR_RGB2YUV}
    if space in allowed_color_space:
        col_conv=cv2.cvtColor(image,allowed_color_space[space])
        return col_conv
    else:
        print('Color space specified doesn\t match the list')
        print('possible choices are bgr,hsv,hls,Lab and gray')
        return None


def img_channel_ext(image,channel=1,show=False):
    '''
    this function extracts given channel from an image
    image: input image
    channel: chhanel to extract in range 0-2
    returns:
        out_ch : output image of the requested channel
    '''
    if ((image.shape[2] !=3) & channel not in list(range(3))):
        print("Err: either input channel is not 3 channel or requested channel number not in range 0-2")
        return -99
    
    out_ch=image[:,:,channel]
    if show:
        fch,axch=plt.subplots()
        axch.imshow(out_ch)              
        axch.set_title("channel"+str(channel)) 
    
    return out_ch  

def apply_color_threshold(image,color_threshold=(0,255),show=False):
    '''
    this function applies threshold on the single channel image
    image: input single channel image
    color_threshold: tuple for low/high thresh
    returns:
        binary_output : thresholded image
    
    '''
    if (len(image.shape) !=2):
        print("Err: input channel >1 , can't apply apply_color_threshold on channel")
        return -99
    
    binary_output = np.zeros_like(image)
    binary_output[(image > color_threshold[0]) & (image <=color_threshold[1])]=1
   
    if show:
        fcthres,axcthres=plt.subplots()
        axcthres.imshow(binary_output)              
        axcthres.set_title("threshold applied to colored channel")
    return  binary_output

#combined thresholded image OR or AND
def img_or(image1, image2,show=False):
    '''
    this function applies bitwise OR on two images, assumed single channel
    soebel grad with threshold or single channel img
    image1: image1
    image2: image2
    returns:
        combined: or'd image
    '''
    if (len(image1.shape) !=2 | len(image2.shape) !=2):
        print("Err:Expected single channel images for OR operation")
        return -99
    combined = np.zeros_like(image1)
    combined[((image1 == 1) | (image2 == 1))] = 1
    
              
    if show:
        fthor,athor=plt.subplots()
        athor.imshow(combined,cmap='gray')
        athor.set_title('OR\'d image')          
    return combined
    
def img_and(image1, image2,show=False):
    '''
    this function applies bitwise AND on two images, assumed single channel
    soebel grad with threshold or single channel img
    image1: image1
    image2: image2
    returns:
        combined: AND'd image
    '''
    if (len(image1.shape) !=2 | len(image2.shape) !=2):
        print("Err:Expected single channel images for AND operation")
        return -99
    combined = np.zeros_like(image1)
    combined[((image1 == 1) & (image2 == 1))] = 1
    
              
    if show:
        fthor,athor=plt.subplots()
        athor.imshow(combined,cmap='gray')
        athor.set_title('AND\'d image')          
    return combined
    
def thresh_and_2color_channels(cim1,cim2,thresh1=(220, 255),thresh2=(190,255),scope=0):
    '''
    this image combines and aplies threshold on 2 color channels
    cim1,cim2: input images
    thresh1,thres2, thresholds to apply
    scope: height of image to consider
    returns:
        combinedm threshold applied on images and AND'd
    in challenge video it may make sense to not consider reduces height on curves
    '''
    #  Apply a threshold to the 1st channel
    cim1_scoped=cim1[scope:,:]
    cim2_scoped=cim2[scope:,:]
             
    cb_output1=apply_color_threshold(cim1_scoped,color_threshold=thresh1,show=False) 
    cb_output2=apply_color_threshold(cim2_scoped,color_threshold=thresh2,show=False)
    combined=img_or(cb_output1, cb_output2,show=False)
    
    return combined
    
def filter_colors_hsv(img,scope=0):
    """
    Convert image to HSV color space and suppress any colors
    outside of the defined color ranges
    """
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow_dark = np.array([15, 127, 127], dtype=np.uint8)
    yellow_light = np.array([25, 255, 255], dtype=np.uint8)
    yellow_range = cv2.inRange(img1, yellow_dark, yellow_light)

    white_dark = np.array([0, 0, 200], dtype=np.uint8)
    white_light = np.array([255, 30, 255], dtype=np.uint8)
    white_range = cv2.inRange(img1, white_dark, white_light)
    yellows_or_whites = yellow_range | white_range
    imgret = cv2.bitwise_and(img1, img1, mask=yellows_or_whites)
    mask2=cv2.cvtColor(imgret,cv2.COLOR_HSV2RGB)
    gray=cv2.cvtColor(mask2,cv2.COLOR_RGB2GRAY)
    mask = cv2.adaptiveThreshold(imgret[:,:,2],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
    
    
    
    return mask

def hls_l_lab_b_img(image,scope=0):
    '''
    this function thresholds and combines hls_l channel and lab b channel
    img: input rgb image
    hls_l_lab_b: output
    '''
    hls=color_space_converter(image,space='hls')
    Lab=color_space_converter(image,space='Lab')
    hls_l=img_channel_ext(hls,channel=1,show=False)
    Lab_b=img_channel_ext(Lab,channel=2,show=False)
    if np.max(Lab_b) > 175:
        Lab_b = Lab_b*(255/np.max(Lab_b))
    hls_l_lab_b=thresh_and_2color_channels(hls_l,Lab_b,thresh1=(200, 255),thresh2=(190,255))
    return hls_l_lab_b[scope:,:]

#for video test
def warp_hsv_y_w(image,scope=0):
    '''
    to test on the video clips if extracting y and w in hsv space
    '''
    undistorted=undistort(image,dist_pickle=dist_pickle)
#    polyimgorig=cv2.polylines(undistorted,[pts],True,(0,0,255),2)
    warp,M,Minv=perspective_trf(undistorted,show=False)
    #extract yellow and white from yuv space
    hsv_y_w=filter_colors_hsv(warp)
    return hsv_y_w[scope:,:]


##test the code
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
    
    imagea=mpimg.imread(imgname11)
    undistorted=undistort(imagea,dist_pickle=dist_pickle)
#    polyimgorig=cv2.polylines(undistorted,[pts],True,(0,0,255),2)
    warp,M,Minv=perspective_trf(undistorted,show=False)
    ## color space channels experiments
    rgb_r = warp[:,:,0]
    rgb_g = warp[:,:,1]
    rgb_b = warp[:,:,2]
    #hsv =
    hsv=color_space_converter(warp,'hsv')
    hsv_h = hsv[:,:,0]
    hsv_s = hsv[:,:,1]
    hsv_v = hsv[:,:,2]
    #Lab = 
    Lab=color_space_converter(warp,'Lab')
    Lab_L = Lab[:,:,0]
    Lab_a = Lab[:,:,1]
    Lab_b = Lab[:,:,2]
    #hsl
    hls=color_space_converter(warp,'hls')
    hls_h=hls[:,:,0]
    hls_l=hls[:,:,1]
    hls_s=hls[:,:,2]
    #yuv
    yuv=color_space_converter(warp,'yuv')
    yuv_y=yuv[:,:,0]
    yuv_u=yuv[:,:,1]
    yuv_v=yuv[:,:,2]
    
    #extract y and w and apply this function to video clip
    warp_y_w=warp_hsv_y_w(imagea)
    #comb of hls_l and lab b channels:
    #hls_l_lab_b=thresh_comb_2color_channels(hls_l,Lab_b,thresh1=(220, 255),thresh2=(190,255),scope=300)
    hls_s_lab_b=thresh_and_2color_channels(hls_s,Lab_b,thresh1=(200, 255),thresh2=(190,255))
    hls_l_lab_b=hls_l_lab_b_img(warp,scope=0)
    yuv_u_v_hls_s = np.stack((yuv_u, yuv_v, hls_s), axis=2)
    yuv_u_v_hls_s_mean = np.mean(yuv_u_v_hls_s, 2)
    
    
    figc, axsc = plt.subplots(6,3)
    figc.subplots_adjust(hspace = 0.5, wspace=0.2)
    axsc = axsc.ravel()
    axsc[0].imshow(rgb_r, cmap='gray')
    axsc[0].set_title('rgb_r', fontsize=10)
    axsc[1].imshow(rgb_g, cmap='gray')
    axsc[1].set_title('rgb_g', fontsize=20)
    axsc[2].imshow(rgb_b, cmap='gray')
    axsc[2].set_title('rgb_b', fontsize=20)
    axsc[3].imshow(hsv_h, cmap='gray')
    axsc[3].set_title('hsv_h', fontsize=20)
    axsc[4].imshow(hsv_s, cmap='gray')
    axsc[4].set_title('hsv_s', fontsize=20)
    axsc[5].imshow(hsv_v, cmap='gray')
    axsc[5].set_title('hsv_v', fontsize=20)
    axsc[6].imshow(Lab_L, cmap='gray')
    axsc[6].set_title('Lab_L', fontsize=20)
    axsc[7].imshow(Lab_a, cmap='gray')
    axsc[7].set_title('Lab_a', fontsize=20)
    axsc[8].imshow(Lab_b, cmap='gray')
    axsc[8].set_title('Lab_b', fontsize=20)
    axsc[9].imshow(hls_h, cmap='gray')
    axsc[9].set_title('hls_h', fontsize=20)
    axsc[10].imshow(hls_l, cmap='gray')
    axsc[10].set_title('hls_l', fontsize=20)
    axsc[11].imshow(hls_s, cmap='gray')
    axsc[11].set_title('hls_s', fontsize=20)    
    axsc[12].imshow(yuv_y, cmap='gray')
    axsc[12].set_title('yuv_y', fontsize=20) 
    axsc[13].imshow(yuv_u, cmap='gray')
    axsc[13].set_title('yuv_u', fontsize=20) 
    axsc[14].imshow(yuv_v, cmap='gray')
    axsc[14].set_title('yuv_v', fontsize=20) 
    axsc[15].imshow(warp_y_w, cmap='gray')
    axsc[15].set_title('warp_y_w', fontsize=20) 
    axsc[16].imshow(hls_l_lab_b, cmap='gray')
    axsc[16].set_title('hls_l_lab_b', fontsize=20)
#    axsc[17].imshow(hls_s_lab_b, cmap='gray')
#    axsc[17].set_title('hls_s_lab_b', fontsize=20)
    axsc[17].imshow(yuv_u_v_hls_s_mean, cmap='gray')
    axsc[17].set_title('yuv_u_v_hls_s_mean', fontsize=20)
    
    



 
#######################################
    
#    vid1 = '../main_code/clip_7_11s.mp4'
#    #vid2 = './clip_22_32s.mp4'
#    #vid3 = './clip_35_47s.mp4'
#    #vid4 = '../main_code/challenge_clip_0_7s.mp4'
#    voutput1='../main_code/clip_7_11s_color_thresh.mp4'
#    #voutput2='./clip_22_32s_proc2.mp4'
#    #voutput3='./clip_35_47s_proc2.mp4'
#    #voutput4='../main_code/challenge_clip_0_7s_color_thresh.mp4'   
#    if os.path.isfile(voutput1) :
#        os.remove(voutput1)
#    #if os.path.isfile(voutput2):
#    #    os.remove(voutput2)
#    #if os.path.isfile(voutput3):
#    #    os.remove(voutput3)
#    #if os.path.isfile(voutput4):
#    #    os.remove(voutput4)    
#    # 
#    video_clip1 = VideoFileClip(vid1)
#    processed_clip1 = video_clip1.fl_image(warp_hsv_y_w)
#    processed_clip1.write_videofile(voutput1, audio=False)
#    ##       
#    #video_clip2 = VideoFileClip(vid2)
#    #processed_clip2 = video_clip2.fl_image(warp_hsv_y_w)
#    #processed_clip2.write_videofile(voutput2, audio=False)
#    #
#    #
#    #video_clip3 = VideoFileClip(vid3)
#    #processed_clip3 = video_clip3.fl_image(warp_hsv_y_w)
#    #processed_clip3.write_videofile(voutput3, audio=False) 
#    #
#    #video_clip4 = VideoFileClip(vid4)
#    #processed_clip4 = video_clip4.fl_image(warp_hsv_y_w)
#    #processed_clip4.write_videofile(voutput4, audio=False)