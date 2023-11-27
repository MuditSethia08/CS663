import argparse
import numpy as np
import scipy.stats as st
from scipy import ndimage
import math
import cv2
import random
import math
import matplotlib.pyplot as plt


def image_to_column(img, w, stepsize=1):
        
    # Parameters
    m, n = img.shape
    col_extent = n - w + 1
    row_extent = m - w + 1

    # Get Starting block indices
    start_idx = np.arange(w)[:, None]*n + np.arange(w)

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*n + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take (img, start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize])


def find_the_best_matches(template, sample_image):
    w = template.shape[0]
    epsilon = 0.1 
    
    # to make a mask that is 1 everhwere the template os filled
    valid_mask = np.zeros_like(template)
    valid_mask = np.where(np.isnan(template), 0, 1)

    # print(valid_mask[: , : , 0].shape)
    
    r_valid_mask = np.reshape(valid_mask[: , : , 0], (-1, 1))
    g_valid_mask = np.reshape(valid_mask[: , : , 1], (-1, 1))
    b_valid_mask = np.reshape(valid_mask[: , : , 2], (-1, 1))
    # print(r_valid_mask.shape)
    

    #to add gaussian
    sigma = w / 6.4
    kernel = cv2.getGaussianKernel(ksize=w, sigma=sigma)
    kernel_2d = kernel * kernel.T

    # normalize mask 
    mask =  kernel_2d / kernel_2d.sum()
    
    vectorized_mask = np.reshape(mask, (-1, 1)) / np.sum(mask)  #vectorize the mask
    
    # partition sample_image to blocks (represented by column vectors)
    
    # (w*w, n_blocks)
    r_sample_image = image_to_column(sample_image[:, :, 0], w)
    g_sample_image = image_to_column(sample_image[:, :, 1], w)
    b_sample_image = image_to_column(sample_image[:, :, 2], w)
    
    n_blocks = r_sample_image.shape[-1]  
    
    # vectorized code that calcualtes SSD(template,sample)*mask for all
    # patches
    
    # (w*w, 1)
    r_temp = np.reshape(template[:, :, 0], (w*w, 1))
    g_temp = np.reshape(template[:, :, 1], (w*w, 1))
    b_temp = np.reshape(template[:, :, 2], (w*w, 1))
    
    # (w*w, n_blocks)
    r_temp = np.tile(r_temp, (1, n_blocks))
    g_temp = np.tile(g_temp, (1, n_blocks))
    b_temp = np.tile(b_temp, (1, n_blocks))
    
    r_dist = vectorized_mask * r_valid_mask * (r_temp - r_sample_image)**2
    g_dist = vectorized_mask * g_valid_mask * (g_temp - g_sample_image)**2
    b_dist = vectorized_mask * b_valid_mask * (b_temp - b_sample_image)**2
    
    # (w*w, n_blocks) -> (n_blocks)
    ssd = np.nansum(np.nansum([r_dist, g_dist, b_dist], axis=0), axis=0)

    # accept all pixel locations whose SSD error values are less than the 
    # minimum SSD value times (1 + ε)
    matches = np.nonzero(ssd)
    errors = ssd[matches]
    min_error = np.min(errors)
    idx = np.where(errors < min_error*(1+epsilon))
    
    best_matches = [matches[0][i] for i in idx[0]]
    errors = [errors[i] for i in idx[0]]
    
    return best_matches, errors


def synthesize(sample_image, w, target_size, implementation = "ours"):
    seed_size = 3
    sample_image = sample_image.astype(float) / 255 #normalization
    
    Channels = sample_image.shape[2]
    
    output_image = np.full((target_size[0], target_size[1], Channels), np.nan)    #the image that is being synthesized
    
    #seed coordinates, we could have picked randomm seed also, NOTE: IDEALLY THIS SHOULD BE THE CENTRE OF THE IMAGE TO AVOID ERRORS
    seed_0=2
    seed_1=2

    #put the seed at the centre of the output image
    c = (int(0.5*target_size[0]) , int(0.5*target_size[1]))   # middle indices of output_image
    
    output_image[c[0]: c[0] + seed_size , c[1]: c[1] + seed_size ,:] = sample_image[seed_0: seed_0 + seed_size , seed_1: seed_1 + seed_size,:]
    
    filled_array = np.zeros(target_size)
    filled_array[c[0]: c[0] + seed_size , c[1]: c[1] + seed_size] = 1  #simce initially the seed is put in the output image
    number_filled = int(np.sum(filled_array))

    how_much_done = 0
    while(number_filled < target_size[0]*target_size[1]):
        how_much_done+=1
        if how_much_done%10==0:
            print("running...")
        
        # to find bordering unfilled pixels:
        new_pixels = np.nonzero(ndimage.binary_dilation(filled_array).astype(filled_array.dtype) - filled_array)
        
        for unfilled_pixel in range(len(new_pixels[0])):
          
            # get indices of the next unfilled pixel
            i, j = new_pixels[0][unfilled_pixel], new_pixels[1][unfilled_pixel]
            padded_im = np.pad(output_image, ((w//2, w//2), (w//2, w//2), (0, 0)), mode='constant', constant_values=np.nan)  #Pad the image with nans to avoid edge cases

            template = padded_im[i:i+w, j:j+w, :]   #the kernel / window / template to be matched with the sample
            
            best_matches, errors = find_the_best_matches(template, sample_image)   #for disha
    
            #  chose the lowest error match OR randomly select from best matches

            if implementation == "theirs":
            
                if len(errors)==1:
                    inex_of_best_best = 0
                else:
                    inex_of_best_best = np.random.randint(0,  len(best_matches)-1)   #randomly select from the best matches

            if implementation == "ours":
                inex_of_best_best = np.argmin(errors)    # different from research paper, select lowest error match
                
            best_match = best_matches[inex_of_best_best]
             
            s_row, s_col = np.unravel_index(best_match, ( sample_image.shape[0] - w + 1,  sample_image.shape[1] - w + 1))   #best_match is linear indices of the pixel of the best match but we need to find the index in the 2d array
            output_image[i, j, :] = sample_image[s_row+w//2, s_col+w//2, :]  # FILL THE PIXEL !!!!!!!
            
            filled_array[i, j] = 1
            number_filled += 1
                

    return output_image

#main
    
image = cv2.imread("color3.png")
image = cv2.resize(image , (54,54))
output_image = synthesize(image, 15, [100, 100])

cv2.imshow("hellow" , output_image)
cv2.waitKey(0)
cv2.imwrite("output.png" , output_image)
