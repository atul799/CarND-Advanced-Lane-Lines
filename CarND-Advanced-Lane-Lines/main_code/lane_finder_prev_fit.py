
"""
this file has function for lane finding when lanes has been found previously with
sliding windows function

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
from sliding_window import *
# function must take already fitted lines produced by "find_lines"
#self,warped,margin=100,minpix=50,show=False
def lanefinder_prev_fit(warped,left_fit, right_fit, margin=100):
    '''
    this function uses input from sliding window function to generate fits for next frames
    
    '''
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = margin
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
                      (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
                       (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
    return left_fit_new, right_fit_new, leftx, lefty,rightx,  righty


#####
#exampleImg2 = cv2.imread('./test_images/test5.jpg')
#exampleImg2 = cv2.cvtColor(exampleImg2, cv2.COLOR_BGR2RGB)
#exampleImg2_bin, Minv = pipeline(exampleImg2) 
#left_fit_new, right_fit_new, leftx,lefty,rightx, righty = lanefinder_prev_fit(img,left_fit, right_fit, margin=margin) 
def visualize_lanefinder_window(img,left_fit,right_fit,leftx,lefty,rightx,righty,margin=100,show=False):  
           
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )    
    # Color in left and right line pixels
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    
    # Create an image to draw on and an image to show the selection window
    out_img = np.uint8(np.dstack((img, img, img))*255)
    window_img = np.zeros_like(out_img)
    
    # Color in left and right line pixels
#    nonzero = img.nonzero()
#    nonzeroy = np.array(nonzero[0])
#    nonzerox = np.array(nonzero[1])
#    out_img[nonzeroy[left_lane_inds2], nonzerox[left_lane_inds2]] = [255, 0, 0]
#    out_img[nonzeroy[right_lane_inds2], nonzerox[right_lane_inds2]] = [0, 0, 255]
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area (OLD FIT)
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    if show:
        plt.imshow(result)
        plt.plot(left_fitx2, ploty, color='yellow')
        plt.plot(right_fitx2, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    return result




#####
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
    imagea=mpimg.imread(imgname4)
    imageb=np.copy(imagea)
    polyimg=cv2.polylines(imageb,[pts],True,(255,0,0),5)
    img_t, Minv = im_pipe(imagea,dist_pickle=dist_pickle,sobel=False,show=False)
    left_fit, right_fit,leftx,lefty,rightx,righty, draw,histogram=sliding_window_polyfit(img_t,margin=100,minpix=50,nwindows=9)
    out_img=visualize_sliding_window(img_t,Minv,left_fit, right_fit, leftx,lefty, rightx,righty, draw,histogram,show=False)
#    imagec=mpimg.imread(imgname6)
#    imaged=np.copy(imagec)
#    polyimg1=cv2.polylines(imaged,[pts],True,(255,0,0),5)
#    img_t1, Minv1 = im_pipe(imagec,dist_pickle=dist_pickle,sobel=False,show=False)
#    left_fit_new, right_fit_new, leftx1, lefty1,rightx1,righty1=lanefinder_prev_fit(img_t1,left_fit, right_fit, margin=100)
#    out_img1=visualize_lanefinder_window(img_t1,left_fit_new,right_fit_new,leftx1,lefty1,rightx1,righty1)
#    f,a=plt.subplots(2,3)
#    a=a.ravel()
#    a[0].imshow(imagea)
#    a[1].imshow(img_t,cmap='gray')
#    a[2].imshow(out_img)
#    a[3].imshow(img_t1)
#    a[4].imshow(out_img1)
#    a[5].plot(histogram)
#    plt.xlim(0, 1280)

   # Set up plot
    fig, axs = plt.subplots(len(img_ar),4, figsize=(25, 30))
    fig.subplots_adjust(hspace = .01, wspace=.1)
    axs = axs.ravel()
    plt.tight_layout()
                  
    i = 0
    for image in img_ar:
        img=mpimg.imread(image)
        imaged=np.copy(img)
        polyimg1=cv2.polylines(imaged,[pts],True,(255,0,0),5)
        img_t1, Minv1 = im_pipe(img,dist_pickle=dist_pickle,sobel=False,show=False)
        left_fit_new, right_fit_new, leftx1, lefty1,rightx1,righty1=lanefinder_prev_fit(img_t1,left_fit, right_fit, margin=100)
        out_img1=visualize_lanefinder_window(img_t1,left_fit_new,right_fit_new,leftx1,lefty1,rightx1,righty1)
        unwarped_img=visualize_sliding_window_image(img,img_t1,Minv,left_fit_new, right_fit_new)
        
        
#        img = mpimg.imread(image)
#        img_t, Minv = im_pipe(img,dist_pickle=dist_pickle,sobel=False,show=False)
#        polyimg=cv2.polylines(img,[pts],True,(255,0,0),5)
#        axs[i].imshow(img)
#        axs[i].axis('off')
#        i += 1
        axs[i].imshow(polyimg1)
        axs[i].axis('off')
        i += 1
        axs[i].imshow(img_t, cmap='gray')
        axs[i].axis('off')
        i += 1 
        axs[i].imshow(out_img1, cmap='gray')
        axs[i].axis('off')
        i += 1
        axs[i].imshow(unwarped_img)
        axs[i].axis('off')
        i += 1