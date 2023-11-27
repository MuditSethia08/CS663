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
    
    epsilon = 0.1

    # Get the dimensions of the template and big image
    h, w = template.shape[:2]
    bh, bw = sample_image.shape[:2]

    min_diff = float('inf')
    min_location = (0, 0)

     
    # to make a mask that is 1 everhwere the template os filled
    valid_mask = np.zeros_like(template)
    valid_mask = np.where(np.isnan(template), 0, 1)

    # print(valid_mask[: , : , 0].shape)
    
    r_valid_mask = valid_mask[: , : , 0]
    g_valid_mask = valid_mask[: , : , 1]
    b_valid_mask = valid_mask[: , : , 2]

  
    #to add gaussian
    sigma = w / 6.4
    kernel = cv2.getGaussianKernel(ksize=w, sigma=sigma)
    kernel_2d = kernel * kernel.T

    # normalize mask 
    mask =  kernel_2d / kernel_2d.sum()

    # Iterate through each possible position in the big image
    
    array_of_errors = []
    array_of_locations = []
    array_of_errors = np.array(array_of_errors)
   

    for i in range(bh - h + 1):
        for j in range(bw - w + 1):

            # Extract the region from the big image
            region_red = sample_image[i:i+h, j:j+w , 0]
            region_green = sample_image[i:i+h, j:j+w , 1]
            region_blue = sample_image[i:i+h, j:j+w , 2]

            r_template = template[:, :, 0]
            g_template = template[:, :, 1]
            b_template = template[:, :, 2]

            # Calculate the squared difference between the template and the region
            
            r_dist = np.nansum(mask * r_valid_mask * np.square(r_template - region_red))
            g_dist = np.nansum(mask * g_valid_mask * np.square(g_template - region_green))
            b_dist = np.nansum(mask * b_valid_mask * np.square(b_template - region_blue))

            diff = r_dist + g_dist +b_dist

            # Update minimum difference and location if a better match is found
            array_of_errors = np.append(array_of_errors , diff)
            array_of_locations.append((i,j))
            if diff < min_diff and diff!=0:
                min_diff = diff
                # min_location = (i, j)


    # print(array_of_locations)
    # print(min_diff)
    idx = np.where(array_of_errors < min_diff*(1+epsilon))
    # print(idx)
    new_array_of_locations = [array_of_locations[i] for i in idx[0]]
    # print(new_array_of_locations)
    new_array_of_errors = [array_of_errors[i] for i in idx[0]]
    # quit()
    # print(new_array_of_errors)

   
    # new_array_of_locations = np.array([min_location])
    # new_array_of_errors = np.array([min_diff])




    # min_error = np.argmin(array_of_errors)
    # new_array_of_errors = []
    # new_array_of_locations = []

    # for i in range(len(array_of_errors)):
    #     if array_of_errors[i] < (1+epsilon)*min_error:

    #         new_array_of_errors.append(array_of_errors[i])
    #         new_array_of_locations.append(array_of_locations[i])

    # new_array_of_errors = np.array(new_array_of_errors)
    # # print(new_array_of_errors)
    
    # new_array_of_locations = np.array(new_array_of_locations)
    
    return new_array_of_locations , new_array_of_errors 



def synthesize(sample_image, w, target_size, implementation = "ours"):
    seed_size = 3
    sample_image = sample_image.astype(float) / 255 #normalization
    
    Channels = sample_image.shape[2]
    
    output_image = np.full((target_size[0], target_size[1], Channels), np.nan)    #the image that is being synthesized
    
    #seed coordinates, we could have picked randomm seed also, NOTE: IDEALLY THIS SHOULD BE THE CENTRE OF THE IMAGE TO AVOID ERRORS
    seed_0=sample_image.shape[0]//2
    seed_1=sample_image.shape[1]//2

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
            # print(errors)
            
    
            #  chose the lowest error match OR randomly select from best matches

            if implementation == "theirs":
            
                if len(errors)==1:
                    inex_of_best_best = 0
                else:
                    inex_of_best_best = np.random.randint(0,  len(best_matches)-1)   #randomly select from the best matches

            if implementation == "ours":
                inex_of_best_best = np.argmin(errors)    # different from research paper, select lowest error match
                
            best_match = best_matches[inex_of_best_best]
             
            s_row, s_col = best_match
            output_image[i, j, :] = sample_image[s_row+w//2, s_col+w//2, :]  # FILL THE PIXEL !!!!!!!
            
            filled_array[i, j] = 1
            number_filled += 1
                

    return output_image

#main
    
image = cv2.imread("color4.png")
image = cv2.resize(image , (34,34))
output_image = synthesize(image, 15, [50, 50])

cv2.imshow("hellow" , output_image)
cv2.waitKey(0)
cv2.imwrite("output.png" , output_image)
