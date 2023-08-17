#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:58:42 2023

@author: zainamoussa
"""
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk

#load images

def align_image(image1,image2):
    #img1_color = cv2.imread(image1)  #use these if file path Image to be aligned.
    #img2_color = cv2.imread(image2) 
    
    #use if img is not black and white already
    #img1 = rgb2gray(image1) #replace with img1_color if file path is used
    #img2 = rgb2gray(image2)
    
    
    img1 = image1
    img2 = image2
    #img2.resize(img1.shape)
    
    #compute optical flow. this tells me the shift in the x and y direction
    v, u = optical_flow_ilk(img1, img2)
    
    #use estimated optical flow for registration
    nr, nc = img1.shape
    
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    
    img2_warp = warp(img2, np.array([row_coords+v, col_coords+u]), mode = 'edge')
    
    #build RGB image with unregistered sequence
    seq_im = np.zeros((nr, nc, 3))
    seq_im[...,0] = img2
    seq_im[...,1] = img1
    seq_im[...,2] = img1
    
    
    
    #build rgb image iwth registered sequence
    reg_im = np.zeros((nr,nc, 3))
    reg_im[...,0]=img2_warp
    reg_im[...,1]=img1
    reg_im[...,2]=img1
    
    target_im = np.zeros((nr, nc, 3))
    target_im[..., 0] = img1
    target_im[..., 1] = img1
    target_im[..., 2] = img1
    
    #show result
    fig, (ax0,ax1,ax2, ax3) = plt.subplots(4,1,figsize=(50,50))
    
    ax0.imshow(seq_im)
    ax0.set_title("Unregistered sequence")
    ax0.set_axis_off()
    
    ax1.imshow(reg_im)
    ax1.set_title("Registered sequence")
    ax1.set_axis_off()
    
    ax2.imshow(target_im)
    ax2.set_title("Target")
    ax2.set_axis_off()
    
    
        # --- Quiver plot arguments
    
    nvec = 20  # Number of vectors to be displayed along each image dimension
    nl, nc = image1.shape
    step = max(nl//nvec, nc//nvec)
    
    y, x = np.mgrid[:nl:step, :nc:step]
    u_ = u[::step, ::step]
    v_ = v[::step, ::step]
    
    norm = np.sqrt(u ** 2 + v ** 2)
    ax3.imshow(norm)
    ax3.quiver(x, y, u_, v_, color='r', units='dots',
               angles='xy', scale_units='xy', lw=3)
    ax3.set_title("Optical flow magnitude and vector field")
    ax3.set_axis_off()
    fig.tight_layout()
    
    plt.show()
    return v,u






