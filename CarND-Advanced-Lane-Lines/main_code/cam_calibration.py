# -*- coding: utf-8 -*-
"""
This package defines function for camera calibration 
on the test images

@author: atpandey
"""
#%%
"""
create chessboard corners, calibrate camera and write pickle with mtx/dst data

"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
#%matplotlib inline
#%matplotlib qt


#%%

def chessboard_corners(images,xc,yc,outdir='../camera_cal/',show=False):
    '''
    this fun finds chessboard corners on a chess image
    images: is input image passed as a list (using glob)
    xc: is number of horizontal corners
    yc: is number of vertical corners
    
    returns:
        objpoints: mpgrid 3D point sets for xc,yc corners
        imgpoints: list of corners found in images
    '''
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
                
    #create objp with mpgrid (0,0,0),1,0,0) ,,,(8,5,0)
    objp=np.zeros((yc*xc,3),np.float32) 
    objp[:,:2]=np.mgrid[0:xc,0:yc].T.reshape(-1,2) #x,y coords
    
        
    fig, axs = plt.subplots(5,4, figsize=(16, 11))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel() 
    i=0
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        print("imagename:",fname,idx)
        img = cv2.imread(fname)
        imgpl = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (xc,yc), None)
        print("retvalchessboard:",ret)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if show:
                # Draw and display the corners
                img=cv2.drawChessboardCorners(imgpl, (xc,yc), corners, ret)
                f2use=fname.split('.')
                write_name = '..'+f2use[2]+'_cornered'+'.jpg'
                print('file name',write_name)
                cv2.imwrite(write_name, img)
                #cv2.imshow('img', img)
                #cv2.waitKey(500)
                #plt.imshow(img)
                axs[i].imshow(img)
                i +=1
    return objpoints,imgpoints

#cv2.destroyAllWindows()
#%%
"""
Calibrate camera and undistort input image

"""



def calibrate_undistort(img, objpoints, imgpoints,pickle_dump_dir='../camera_cal/'):
    '''
    this function calibrates the camera and dumps the camera matrix(mtx) and
    distortion (dist) matirx into a picke file
    img: is image input
    objpoints: is the list of mpgrid functions for the corners in chess board
    imgpoints: is list of  2D points found by findChessboardCorners functions for
             images about 17 examples provided by udacity git repo
    pickle_dump_dir: directory where mtx and dist matrix data is to be saved
    
    return:
        dst : undistorted image
    '''
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    #cv2.imwrite('test_images/test1_undist.jpg',dst)
    
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle_dum_f=pickle_dump_dir+'/calib_pickle.p'
    pickle.dump( dist_pickle, open( pickle_dum_f, 'wb') )
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    return dst

#%%

#unittest
if __name__=='__main__':
    # Make a list of calibration images
    images = glob.glob('../camera_cal/*.jpg')
    
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    #horizontal corners in chess board
    x_corners=9
    #vertical corners on chessboard
    y_corners=6

    objpoints,imgpoints=chessboard_corners(images,x_corners,y_corners,show=True)


    # Test undistortion on an image
    img_2_undist='../camera_cal/calibration1.jpg'
    imgt = cv2.imread(img_2_undist)
    #imgt=cv2.cvtColor(imgt,cv2.COLOR_BGR2RGB)
    img_size = (imgt.shape[1], imgt.shape[0])
    undist_image=calibrate_undistort(imgt, objpoints, imgpoints,pickle_dump_dir='../camera_cal/')
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(imgt)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(undist_image)
    ax2.set_title('Undistorted Image', fontsize=30)
    write_name = '../camera_cal/'+'/calibration1_undistorted'+'.jpg'
    print('file name',write_name)
    cv2.imwrite(write_name, undist_image)