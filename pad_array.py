#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:41:28 2023

@author: zainamoussa
"""
from PIL import Image
import numpy as np
import cv2
import scipy
from skimage.color import rgb2gray
from scikit_image_align import align_image 
import random
from phase_cross_correlation import cross_correlate
import matplotlib.pyplot as plt

def pad_image_with_zeros(image_array,pad_size): #pad_size is number of zeros to add in between actual info
    x,y = image_array.shape
    empty_array = np.zeros([(pad_size+1)*x, (pad_size+1)*y]) #creates new array of zeros
    row, col =empty_array.shape
    for i in range(x):#for each row in the array
        for j in range(y): #for each column in the row
            empty_array[i*(pad_size+1)][j*(pad_size+1)]= image_array[i][j]
    '''padded_array = np.empty((empty_array.shape[0],empty_array.shape[1]), dtype = object)
    for i in range(x):#for each row in the array
        for j in range(y): #for each column in the row
            padded_array[i][j]= [empty_array[i][j]][0] 
            '''
    return empty_array

def make_sinc_grid(grid_size, pad_size):
    #make sinc grid
    x = np.linspace(-grid_size/2, grid_size/2, grid_size, dtype = int)
    y = np.linspace(-grid_size/2, grid_size/2, grid_size, dtype = int)
    # full coordinate arrays
    
    count = 0
    sinc_array = np.zeros([grid_size, grid_size]) 
    for i in range(len(x)):
        for j in range(len(y)):
            sinc_array[i][j] = np.sinc(x[i]/(pad_size+1))*np.sinc(y[j]/(pad_size+1))
            count+= sinc_array[i][j]               
    sinc_array = sinc_array/count #divide by sum to normalize
    newcount = 0
    for i in range(len(x)):
        for j in range(len(y)):
            newcount+= sinc_array[i][j]

    sinc_img = Image.fromarray(sinc_array/(np.amax(sinc_array))*255)       
    
    return sinc_array, sinc_img

def convolve_and_crop(sinc_array, padded_array, sinc_grid_size):
    convolved_array = scipy.signal.convolve(sinc_array,padded_array)
    crop_amt =int((sinc_grid_size/2))
    cropped_norm_conv = convolved_array[crop_amt-1:len(convolved_array)-crop_amt, crop_amt-1:len( convolved_array)-crop_amt]
    normalized_conv = cropped_norm_conv/(np.amax(cropped_norm_conv))*255
    convolved_img = Image.fromarray(normalized_conv)
    return cropped_norm_conv, convolved_img


def downsample_img(cropped_img,convolved_array, pad_size, grab_random_shift): #hmmm how do i gurantee it starts at 1? cant crop exactly
    down_array = np.zeros([cropped_img.shape[0], cropped_img.shape[1]]) 
    grab_random_shift = grab_random_shift#random.randint(1,pad_size) #randomly set the shifted pixels to grab
    icount = 0
    for i in range(0,len(convolved_array)-1,pad_size+1):
        jcount = 0
        for j in range(0, len(convolved_array)-1, pad_size+1):
            down_array[icount][jcount] = convolved_array[i+grab_random_shift][j]
            jcount+=1
        icount+=1
    down_img = Image.fromarray(down_array/(np.amax(down_array))*255)       
    return down_array, down_img
    
    


#---------------------------------------------
img1read = cv2.imread("Pos0.jpg")
img1 = rgb2gray(img1read) 
cropped_img = img1[0:1024,0:1024]
'''
pad_size = 100

padded_array= pad_image_with_zeros(cropped_img,pad_size)
cropped_array = cropped_img/(np.amax(cropped_img))*255

og_img = Image.fromarray(cropped_img/(np.amax(cropped_img))*255) #n ormalize to 0,255
#og_img.show()

padded_image = Image.fromarray(padded_array/(np.amax(padded_array))*255)
#padded_image.show()

sinc_grid_size = 256
sinc_array, sinc_img = make_sinc_grid(sinc_grid_size, pad_size)
#sinc_img.show()
convolved_array, convolved_img = convolve_and_crop(sinc_array, padded_array, sinc_grid_size)
#convolved_img.show()
'''
'''
import csv 
with open('Upsample_factors_MSE.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(["Pad Size", "MSE","1/Pad Size" ])

with open('Upsampled Shifts.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Pad size", "Generated Shift", "Detected Shift"])
 '''   
figure, axis = plt.subplots(1, 2)

mean_sq_error=[]
computed_pads = []
pad_spacing = [] 
keep = []
throw = []
pad_size = range(1,25,1)
for i in pad_size: #for each given pad size create the image to upsample
    padded_array= pad_image_with_zeros(cropped_img,i)
    cropped_array = cropped_img/(np.amax(cropped_img))*255
    og_img = Image.fromarray(cropped_img/(np.amax(cropped_img))*255) #normalize to 0,255
    #og_img.show()
    padded_image = Image.fromarray(padded_array/(np.amax(padded_array))*255)
    #padded_image.show()
    sinc_grid_size = int(padded_array.shape[0]*1/8) #128
    sinc_array, sinc_img = make_sinc_grid(sinc_grid_size, i)
    print("padded image shape", padded_array.shape)
    print("sinc array shape", sinc_array.shape)
    #sinc_img.show()
    convolved_array, convolved_img = convolve_and_crop(sinc_array, padded_array, sinc_grid_size)
    print ("finished convolving for pad size", i)
    shift = []
    detected_xshift = []
    mean_square_error = []
    y1=[]
    move =[]
    for j in range(i+1): #for each upsampled image downsample and determine shift
        downsampled_array, downsampled_img = downsample_img(cropped_img,convolved_array, i,j)
        #downsampled_img.show()
        downsample_array = downsampled_array/(np.amax(downsampled_array))
        #phase cross correlation algorithm
        sub_shift, sub_error, sub_diffphase = cross_correlate(cropped_img, downsample_array)
        shift.append(j/(i+1))
        detected_xshift.append(sub_shift[0])
        MSE = np.square(np.subtract(shift,detected_xshift))
        
        
        move.append(0-sub_shift[0])
        y1.append(sub_shift[0]+move[0])
        '''
        with open('Upsampled Shifts.csv', 'a+', newline='') as file:
            writer = csv.writer(file)    
            writer.writerow([i, j/(i+1), sub_shift[0]])
        '''
    print("finished detecting shift for pad size", i)
    x1 = shift
    
    #move = 0-detected_xshift[0]
    #y1 = detected_xshift
    #for i in range(len(y1)):
    #    y1[i]=y1[i]+move
    
    compare_x = np.linspace(0,1,i+1)
    compare_y = compare_x
  
    axis[0].plot(x1,y1,label='line %s' %j)
    #axis[0].plot(compare_x[0:-1],compare_y[0:-1], label = 'reference line')
    #axis[0].xlabel('generated shift')
    #axis[0].ylabel('detected shift')
    axis[0].set_title('Detected shifts based on pad_size')   
    axis[0].legend 

    MSE = np.square(np.subtract(compare_y,detected_xshift)).mean()    
    mean_sq_error.append(MSE)
    computed_pads.append(i)
    pad_spacing.append(1/i)
    #print(i, MSE)
    if MSE > 1/i:
        throw.append([i,MSE])
    else:
        keep.append([i,MSE])
    '''
    with open('Upsample_factors_MSE.csv', 'a+', newline='') as file:
        writer = csv.writer(file)       
        writer.writerow([i, MSE, 1/i ])
    '''
    

        
'''

axis[1].plot(computed_pads,mean_sq_error, label = 'MSE')
axis[1].plot(computed_pads,pad_spacing, label = 'acceptable spacing')
axis[1].set_title('MSE of varying pad sizes')
'''
plt.plot(computed_pads,mean_sq_error, label = 'MSE')
plt.plot(computed_pads,pad_spacing, label = 'acceptable spacing')
plt.title('MSE of varying pad sizes')
plt.xlabel('Pad Size')
plt.ylabel('Mean Squared Error')




#print('Shift:',shift)
#print('Detected_shift', detected_xshift)







'''optical flow algorithm
#input downsample image and og image here to compute optical shift
v,u = align_image(cropped_img, downsample_array) '''
