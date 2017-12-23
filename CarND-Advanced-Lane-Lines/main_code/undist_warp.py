
"""
this module implements def for undistort and warp an image
@author: atpandey
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle


def undistort(image,dist_pickle,show=False):
    '''
    this function implements undistort of image
    image: input image
    dist_pickle: pickle data
    returns:
        undist : undistotrted data
    '''
    mtx=dist_pickle['mtx'] 
    dist=dist_pickle['dist'] 
    undist1 = cv2.undistort(image, mtx, dist, None, mtx)
    #aaply gaussian blur
    undist=cv2.GaussianBlur(undist1, (3, 3), 0)
    #self.undistorted=undist
    if show:
        fud,axud=plt.subplots()
        axud.imshow(undist)
        axud.set_title("Undistorted image")
    
    return undist
    
#exampleImg_undistort = undistort(exampleImg)
#
## Visualize undistortion
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#f.subplots_adjust(hspace = .2, wspace=.05)
#ax1.imshow(exampleImg)
#ax1.set_title('Original Image', fontsize=30)
#ax2.imshow(exampleImg_undistort)
#ax2.set_title('Undistorted Image', fontsize=30)


def perspective_trf(image,show=False):
    
    '''
    this fuction performs perspective transform and return warped image and M,Minv matrix
    image:input image
    src: np array of four point on input image (leftbottom,lefttop,righttop,rightbottom)
    dest: np array of four points on warped image should be in order as src points
    returns:
        warped: warped image
        M: pers matrix
        Minv: inv pers matrix
    '''

#    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
#    #corners = np.float32([[190,720],[550,480],[710,480],[1145,720]])
##    corners = np.float32([[220,680],[550,480],[710,480],[1045,680]])
#    new_top_left=np.array([corners[0,0],0])
#    new_top_right=np.array([corners[3,0],0])
#    offset=[150,0]
#    
#    img_size = (image.shape[1], image.shape[0])
#    src = np.float32([corners[0],corners[1],corners[2],corners[3]])
#    dest = np.float32([corners[0]+offset,new_top_left+offset,new_top_right-offset ,corners[3]-offset])
    
    src = np.float32([[696, 455],
                 [1096, 719],
                 [206, 719],
                 [587, 455]])

    dest = np.float32([[930, 0],
                 [930, 719],
                 [350, 719],
                 [350, 0]])
    
    
    
    
    M     = cv2.getPerspectiveTransform(src, dest)
    Minv = cv2.getPerspectiveTransform(dest, src)
    warp=cv2.warpPerspective(image, M, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    #warp=cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    
    if show:
        fp,axp=plt.subplots()
        axp.imshow(warp)
        axp.set_title("Perspective transformed image")
    return warp,M,Minv




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
    
    baseimg_count=0
    challenge_count=0
    llen_list=len(img_ar)
    fpli,axpli=plt.subplots(4,3,figsize=(25,10))
    fpli.subplots_adjust(hspace = .05, wspace=.2)
    axpli=axpli.ravel()
    fpli11,axpli11=plt.subplots(4,3,figsize=(25,10))
    fpli11.subplots_adjust(hspace = .05, wspace=.2)
    axpli11=axpli11.ravel()
    fpli1,axpli1=plt.subplots(3,3,figsize=(25,10))
    fpli1.subplots_adjust(hspace = .05, wspace=.2)
    axpli1=axpli1.ravel()
    
    
    i=0
    j=0
    k=0
    
    # source and destination points for warp transform

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
    
#    imagea=mpimg.imread(imgname1)
#    undistorted=undistort(imagea,dist_pickle=dist_pickle)
#    polyimgorig=cv2.polylines(undistorted,[pts],True,(0,0,255),2)
#    warped,M,Minv=perspective_trf(undistorted,src=src,dest=dst,show=False)
#    fpli2,axpli2=plt.subplots(1,3,figsize=(25,10))
#    fpli2.subplots_adjust(hspace = .05, wspace=.1)
#    axpli2=axpli2.ravel()
#    axpli2[0].imshow(undistorted)
#    axpli2[1].imshow(polyimgorig)
#    axpli2[2].imshow(warped)
    
#######################################
    
    for imagen in img_ar:
        print("image being processed is",imagen)
        image = mpimg.imread(imagen)
    
        img_width=image.shape[1]
        img_height=image.shape[0] 
        
        offset = 50


        
        undistorted=undistort(image,dist_pickle=dist_pickle)
        if baseimg_count < 4:
            axpli[i].imshow(undistorted)
            if baseimg_count==0:
                axpli[i].set_title("undistorted image")
            i +=1
        elif baseimg_count >= 4 and baseimg_count < 8:
            axpli11[k].imshow(undistorted)
            if baseimg_count==4:
                axpli11[k].set_title("undistorted image")
            k +=1
        else:
            axpli1[j].imshow(undistorted)
            if baseimg_count==8:
                axpli1[j].set_title("undistorted image")
            j +=1
        #warped,M,Minv=perspective_trf(undistorted,src=src,dest=dest,show=True)
        
#        pts=np.int32(src)
#        pts = pts.reshape((-1,1,2))
        polyimgorig=cv2.polylines(undistorted,[pts],True,(0,0,255),2)
        
        
        warped,M,Minv=perspective_trf(polyimgorig,show=False)


        if baseimg_count < 4:
            axpli[i].imshow(polyimgorig)
            if baseimg_count==0:
                axpli[i].set_title("polygon drawn")
            i +=1
            axpli[i].imshow(warped)
            if baseimg_count==0:
                axpli[i].set_title("warped poly")
            i +=1
        elif baseimg_count >= 4 and baseimg_count < 8:
            axpli11[k].imshow(polyimgorig)
            if baseimg_count==4:
                axpli11[k].set_title("polygon drawn")
            k +=1
            axpli11[k].imshow(warped)
            if baseimg_count==4:
                axpli11[k].set_title("warped poly")
            
            k +=1
        else:
            axpli1[j].imshow(polyimgorig)
            if baseimg_count==8:
                axpli1[j].set_title("polygon drawn")
            j +=1
            axpli1[j].imshow(warped)
            if baseimg_count==8:
                axpli1[j].set_title("warped poly")
            j +=1
        
        baseimg_count +=1