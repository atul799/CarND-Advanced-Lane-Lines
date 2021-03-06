# -*- coding: utf-8 -*-
"""
Line Class is defined to keep track of prev  detected lanes
best_fit is used as an average of 5 prev fits to keep lane lines smooth between frames

@author: atpandey
"""
import cv2
import numpy as np

class Line():
    '''
    this class defines quite a few attributes to keep track of data on left/roght lanes
    very few used in this implementation
    
    '''
    
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None  
        
        #poly coeff of last n iters
        self.recent_fits=[]
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #number of detected pixels
        self.px_count = None
        self.num_missed = 0

                
    def update_fit(self, fit):    
        '''make detected true if fit is found
        average current_fits and assign to best fit
        '''
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
            ##not very effective idea
            if (self.diffs[0] > 0.001 or \
               self.diffs[1] > 1.0 or \
               self.diffs[2] > 100.) and \
               len(self.current_fit) > 0:
                # bad fit
                self.detected = False
            else:
                self.detected = True
#                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                if len(self.current_fit) > 8:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[-8:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)