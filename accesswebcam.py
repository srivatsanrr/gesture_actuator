# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:10:14 2018

@author: Srivatsan
"""

import cv2 

def open_webcam(mirror=False):
    camera= cv2.VideoCapture(0)
    while True:
        retval, img=camera.read() ## read camera data and store in img
        if (mirror):
            img=cv2.flip(img, 1) ##Lateral Inversion 
        cv2.imshow('The I-Eye', img) ##display img
        if cv2.waitKey(1)==27: ## 27 is ASCII for esc key 
            break
    cv2.destroyAllWindows()

def main():
    open_webcam(mirror=True)

if __name__=='__main__':
    main()
