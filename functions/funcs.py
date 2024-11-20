import pandas as pd
import numpy as np
import random
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from functions.classifiers3x3 import c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13
from functions.feedback import KMeansSegmentation, VectorSegmenter, HyperplaneSegmentation

#############################################################################
#############################################################################
"""
    Returns an array of column numbers representing the pixels that make up a kxk square 
    with input pixel `p` as the top-left corner, for an image of width `w` and height `h`.
    
    Parameters:
    - p: int, index of the top-left corner pixel in the flattened array
    - w: int, width of the image
    - h: int, height of the image
    - k: int, size of the square (defaults to 3x3)
    
    Returns:
    - List[int]: A list of indices representing the pixels in the kxk square
"""

def get_kxk_square(p:int, k:int=3, w:int=28, h:int=28):
    
    # Calculate the row and column of the top-left pixel `p`
    row = p // w
    col = p % w
    
    # Ensure the square fits within the bounds of the image
    if row + k > h or col + k > w:
        raise ValueError(f"A {k}x{k} square out of range for index p={p}.")
    
    # Store the indices
    square_indices = []
    
    # Loop through each pixel in the kxk square
    for i in range(k):
        for j in range(k):
            # Calculate the flattened index for each pixel in the square
            square_index = (row + i) * w + (col + j)
            square_indices.append(square_index)
    
    return square_indices

#############################################################################
#############################################################################
"""
    Returns the values in row `r` at the indices specified in `square_pixels`,
    arranged in a kxk grid format.
    
    Parameters:
    - r: List[int] or array-like, representing the row of data (e.g., a flattened image).
    - pix: List[int], indices of a kxk square within `r`.
    - data: pd.DataFrame, the dataset of all numbers that we care about.
    
    Returns:
    - List[int]: A list of values representing the pixel values in the kxk square
"""

def get_kxk_square_values(r:int, k:int, pix:list[int], data:pd.DataFrame):
    row = data.iloc[r]

    # Get the values from row r at the indices specified in square_pixels
    values = [row.iloc[idx] for idx in pix]
    values = np.array(values).reshape(k, k)
    return values

#############################################################################
#############################################################################
"""
    Assess a kxk square for dark and light rows and columns, and return if it is interesting.
    
    Parameters:
    - values_list: List[int] or array-like, length 9, representing a 3x3 square of pixels.

    Returns:
    - dark_cols: bool, True if dark-masked columns exist, False otherwise.
    - dark_rows: bool, True if dark-masked rows exist, False otherwise.
    - light_cols: bool, True if light-masked columns exist, False otherwise.
    - light_rows: bool, True if light-masked rows exist, False otherwise.
    - is_interesting: bool, True if any of dark_cols, dark_rows, light_cols, light_rows are true, False otherwise.
"""

def assess_square(values_list: list[int]):
    # Calculate the average value and the thresholds for "dark" and "light"
    avg_value = np.mean(values_list)
    dark_threshold = avg_value + 40
    light_threshold = avg_value - 40
    
    # Define the binary masks for "dark" and "light"
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # Check for dark / light cols / rows
    dark_cols = np.any(np.all(dark_mask, axis=0))
    dark_rows = np.any(np.all(dark_mask, axis=1))
    light_cols = np.any(np.all(light_mask, axis=0))
    light_rows = np.any(np.all(light_mask, axis=1))

    is_interesting = dark_cols or dark_rows or light_cols or light_rows

    return avg_value, dark_cols, dark_rows, light_cols, light_rows, is_interesting

#############################################################################
#############################################################################
"""
    Return the binarized results of all 14 classifiers for each 3x3 square of pixels in an image.
    
    Parameters:
    - values_list: List[int] or array-like, length 9, representing a 3x3 square of pixels
    - data: pd.DataFrame, the dataset of all numbers, each row being a distinct number
    - image: int, the specific image (row # of 'data') to be analyzed
    - n: int, the number of total pixels in the image (defaults to 784)
    - w: int, width of the image (defaults to 28)
    - h: int, height of the image (defaults to 28)
    - k: int, size of the square (defaults to 3x3)

    Returns:
    - binary_matrix: np.array, boolean output of each sensor, converted to binary
"""

def signal_results(data:pd.DataFrame, image:int, n:int=784, k:int=3, w:int=28, h:int = 28):
    all_sigs = []
    for i in range(n):
        row = i // w
        col = i % w
        if not (row + k > h or col + k > w):
            pix = get_kxk_square(p = i, k = k)
            values_list = get_kxk_square_values(r = image, k = k, pix = pix, data = data)
            signals = assess_square(values_list = values_list)
            all_sigs.append([c0(signals), c1(signals)
                            , c2(values_list, signals), c3(values_list, signals), c4(values_list, signals), c5(values_list, signals), c6(values_list, signals), c7(values_list, signals)
                            , c8(values_list, signals), c9(values_list, signals), c10(values_list, signals), c11(values_list, signals), c12(values_list, signals), c13(values_list, signals)])

    all_sigs = np.array(all_sigs) 
    binary_matrix = all_sigs.astype(int)
    return(binary_matrix)

#############################################################################
#############################################################################
"""
    Aggregates signals of similar types by row.
    
    Parameters:
    - binary_matrix: np.array, boolean otput of each sensor, converted to binary
    - n: int, the number of total pixels in the image (defaults to 784)
    - w: int, width of the image (defaults to 28)
    - h: int, height of the image (defaults to 28)
    - k: int, size of the square (defaults to 3x3)

    Returns:
    - result_vector: list, sum of all sensor outputs by sensor type by row
"""

def row_agg(binary_matrix:np.array, n:int=784, k:int=3, w:int=28, h:int = 28):
    # Reshape to aggregate different sensor types along rows (change this up for different aggregation)
    reshaped_array = binary_matrix.reshape(w-(k-1), binary_matrix.shape[1], -1)

    # Sum along the reshaped first axis to get a (14, 26) matrix with block sums
    block_sums = reshaped_array.sum(axis=0)

    # Flatten the matrix to create a 392-length vector, representing all of the different aggregation points
    result_vector = block_sums.flatten()
    return(result_vector)

    
#############################################################################
#############################################################################
"""
    Initialize the different plane segmentors
    
    Parameters:
    - num_sections: int, number of segments for output space
    - len: int, length out output space (defaults to 364)

    Returns:
    - kmeans_segmenter: object, kmeans segmenter
    - plane_segmenter: object, plane segmenter
"""

def initialize_segmenters(num_sections:int, len:int=364):
    # Generate 10000 random binary vectors
    sample_vectors = np.random.randint(0, 2, (10000, len))

    # Using KMeansSegmentation strategy
    kmeans_strategy = KMeansSegmentation(num_sections)
    kmeans_segmenter = VectorSegmenter(strategy=kmeans_strategy)
    kmeans_segmenter.fit(sample_vectors)
    # print("KMeans Segment:", kmeans_segmenter.assign_segment(output))

    # Switching to HyperplaneSegmentation strategy
    hyperplane_strategy = HyperplaneSegmentation(num_sections)
    plane_segmenter = VectorSegmenter(strategy=hyperplane_strategy)
    plane_segmenter.set_strategy(hyperplane_strategy)
    plane_segmenter.fit(sample_vectors)
    # print("Hyperplane Segment:", plane_segmenter.assign_segment(output))

    return kmeans_segmenter, plane_segmenter

#############################################################################
#############################################################################
"""
    Completes a full step from image input to kmeans_amps update.
    
    Parameters:
    - values_list: List[int] or array-like, length 9, representing a 3x3 square of pixels
    - data: pd.DataFrame, the dataset of all numbers, each row being a distinct number
    - image: int, the specific image (row # of 'data') to be analyzed
    - amps: list[int], the amplifiers for each aggregated set of signals
    - labels: pd.Series, the correct number label for that training image
    - n: int, the number of total pixels in the image (defaults to 784)
    - num_sections: int, distinct segments of output space

    Returns:
    - kmeans_amps: list, updated amplifier based on feedback
    - kmeans_correct: int, 0 if not a match, 1 if match
"""

def kmeans_step(data:pd.DataFrame, image:int, kmeans_amps:list[int], 
         labels:pd.Series, kmeans_segmenter, n:int=784, num_sections:int=15):

    print(f"-----------------IMAGE {image}-----------------")

    real = labels[image]

    binary_matrix = signal_results(data=data, image=image)

    result_vector = row_agg(binary_matrix=binary_matrix)

    kmeans_output = result_vector + kmeans_amps
    kmeans_output_binary = (kmeans_output > 5).astype(int)

    kmeans_section = kmeans_segmenter.assign_segment(kmeans_output_binary)

    if kmeans_section == real:
        print(f"kmeans match: guess = {kmeans_section} , actual = {real}")
        kmeans_amps[result_vector > 0] += 1
        kmeans_correct = 1
    else: 
        print(f"kmeans no match: guess = {kmeans_section} , actual = {real}")
        kmeans_amps[result_vector == 0] = np.round(kmeans_amps[result_vector == 0] / 1.5, 1)
        kmeans_correct = 0

    return kmeans_amps, kmeans_correct

#############################################################################
#############################################################################
"""
    Completes a full step from image input to plane_amps update.
    
    Parameters:
    - values_list: List[int] or array-like, length 9, representing a 3x3 square of pixels
    - data: pd.DataFrame, the dataset of all numbers, each row being a distinct number
    - image: int, the specific image (row # of 'data') to be analyzed
    - amps: list[int], the amplifiers for each aggregated set of signals
    - labels: pd.Series, the correct number label for that training image
    - n: int, the number of total pixels in the image (defaults to 784)
    - num_sections: int, distinct segments of output space

    Returns:
    - plane_amps: list, updated amplifier based on feedback
    - plane_correct: int, 0 if not a match, 1 if match
"""

def plane_step(data:pd.DataFrame, image:int, plane_amps:list[int], 
         labels:pd.Series, plane_segmenter, n:int=784, num_sections:int=15):

    print(f"-----------------IMAGE {image}-----------------")

    real = labels[image]

    binary_matrix = signal_results(data=data, image=image)

    result_vector = row_agg(binary_matrix=binary_matrix)

    plane_output = result_vector + plane_amps
    plane_output_binary = (plane_output > 5).astype(int)

    plane_section = plane_segmenter.assign_segment(plane_output_binary)

    if plane_section == real:
        print(f"plane match: guess = {plane_section} , actual = {real}")
        plane_amps[result_vector > 0] += 1
        plane_correct = 1
    else: 
        print(f"plane no match: guess = {plane_section} , actual = {real}")
        plane_amps[result_vector == 0] = np.round(plane_amps[result_vector == 0] / 1.5, 1)
        plane_correct = 0

    return plane_amps, plane_correct
