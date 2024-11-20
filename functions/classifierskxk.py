import numpy as np
import pandas as pd
from functions.funcs import get_kxk_square, get_kxk_square_values, assess_square
from functions.feedback import KMeansSegmentation, VectorSegmenter, HyperplaneSegmentation

def bars(signals: list, values_list: list[int]):
    k = len(values_list)
    nleft = np.floor(k / 2).astype(int)
    nright = np.ceil(k / 2).astype(int)

    avg = signals[0]
    dark_cols = signals[1]
    dark_rows = signals[2]
    light_cols = signals[3]
    light_rows = signals[4]
    is_interesting = signals[5]

    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    classifications = []

    count = 0

    if not is_interesting:
        if avg < 100:
            classifications.append('c0')
        else:
            classifications.append('c1')
    else:
        if dark_cols or light_cols:
            if (
                    any(np.all(r) for r in [dark_mask[:,:n+1].tolist() for n in range(nleft)])  # Dark left
                    or
                    any(np.all(r) for r in [light_mask[:, 1+n:].tolist() for n in range(nleft)])  # Light middle and right
                ):
                    classifications.append('c2')

            if (
                any(np.all(r) for r in [light_mask[:, -(n+1):].tolist() for n in range(nleft)])  # Light right
                or
                any(np.all(r) for r in [dark_mask[:,:nleft+n+1].tolist() for n in range(nright-1)])  # Dark left and middle
            ):
                classifications.append('c3')

            if (
                (
                    any(np.all(r) for r in [dark_mask[:,:n+1].tolist() for n in range(nleft)])  # Dark left
                    and
                    any(np.all(r) for r in [dark_mask[:, -(n+1):].tolist() for n in range(nleft)])  # Dark right
                )
                and
                np.any([np.all([np.all(r) for r in objs]) for objs in [light_mask[:, n+1:m+1].tolist() for n in range(k-1) for m in range(n + 1, k-1)]])  # Light middle
            ):
                classifications.append('c4')

            if (
                (
                    any(np.all(r) for r in [light_mask[:,:n+1].tolist() for n in range(nleft)])  # Light left
                    and
                    any(np.all(r) for r in [light_mask[:, -(n+1):].tolist() for n in range(nleft)])  # Light right
                )
                and
                np.any([np.all([np.all(r) for r in objs]) for objs in [dark_mask[:, n+1:m+1].tolist() for n in range(k-1) for m in range(n + 1, k-1)]])  # Dark middle
            ):
                classifications.append('c5')

            if (
                any(np.all(r) for r in [light_mask[:,:n+1].tolist() for n in range(nleft)])  # Light left
                or
                any(np.all(r) for r in [dark_mask[:, 1+n:].tolist() for n in range(nleft)])  # Dark middle and right
            ):
                classifications.append('c6')

            if (
                any(np.all(r) for r in [dark_mask[:, -(n+1):].tolist() for n in range(nleft)])  # Dark right
                or
                any(np.all(r) for r in [light_mask[:,:nleft+n+1].tolist() for n in range(nright-1)])  # Light left and middle
            ):
                classifications.append('c7')

            # If nothing has been added to classifications but it is intersting, check if dark or light
            if len(classifications) == 0 and (not (dark_rows or light_rows)):
                if avg < 100:
                    classifications.append('c0')
                else:
                    classifications.append('c1')

        ####################### Horizontal Bar Conditions #######################

        if dark_rows or light_rows:
            if (
                    any(np.all(r) for r in [dark_mask[:n+1,:].tolist() for n in range(nleft)])  # Dark top
                    or
                    any(np.all(r) for r in [light_mask[1+n:, :].tolist() for n in range(nleft)])  # Light middle and bottom
                ):
                    classifications.append('c8')

            if (
                any(np.all(r) for r in [light_mask[-(n+1):, :].tolist() for n in range(nleft)])  # Light bottom
                or
                any(np.all(r) for r in [dark_mask[:nleft+n+1,:].tolist() for n in range(nright-1)])  # Dark top and middle
            ):
                classifications.append('c9')

            if (
                (
                    any(np.all(r) for r in [dark_mask[:n+1,:].tolist() for n in range(nleft)])  # Dark top
                    and
                    any(np.all(r) for r in [dark_mask[-(n+1):, :].tolist() for n in range(nleft)])  # Dark bottom
                )
                and
                np.any([np.all([np.all(r) for r in objs]) for objs in [light_mask[n+1:m+1, :].tolist() for n in range(k-1) for m in range(n + 1, k-1)]])  # Light middle
            ):
                classifications.append('c10')

            if (
                (
                    any(np.all(r) for r in [light_mask[:n+1,:].tolist() for n in range(nleft)])  # Light top
                    and
                    any(np.all(r) for r in [light_mask[-(n+1):, :].tolist() for n in range(nleft)])  # Light bottom
                )
                and
                np.any([np.all([np.all(r) for r in objs]) for objs in [dark_mask[n+1:m+1, :].tolist() for n in range(k-1) for m in range(n + 1, k-1)]])  # Dark middle
            ):
                classifications.append('c11')

            if (
                any(np.all(r) for r in [light_mask[:n+1,:].tolist() for n in range(nleft)])  # Light top
                or
                any(np.all(r) for r in [dark_mask[1+n:, :].tolist() for n in range(nleft)])  # Dark middle and bottom
            ):
                classifications.append('c12')

            if (
                any(np.all(r) for r in [dark_mask[-(n+1):, :].tolist() for n in range(nleft)])  # Dark bottom
                or
                any(np.all(r) for r in [light_mask[:nleft+n+1,:].tolist() for n in range(nright-1)])  # Light top and middle
            ):
                classifications.append('c13')

            # If nothing has been added to classifications but it is intersting, check if dark or light
            if len(classifications) == 0:
                if avg < 100:
                    classifications.append('c0')
                else:
                    classifications.append('c1')

    return classifications

#############################################################################
#############################################################################
"""
    Return the list of activated sensor types for each kxk square present in the image (as read in normal reading order)
"""

def all_kxk_activations(data:pd.DataFrame, image:int, n:int=784, k:int=3, w:int=28, h:int = 28):
    all_activs = []
    for i in range(n):
        row = i // w
        col = i % w
        if not (row + k > h or col + k > w):
            pix = get_kxk_square(p = i, k = k)
            values_list = get_kxk_square_values(r = image, k = k, pix = pix, data = data)
            signals = assess_square(values_list=values_list)
            activations = bars(signals=signals, values_list=values_list)
            all_activs.append(activations)
    # binary_matrix = all_sigs.astype(int)
    return(all_activs)

#############################################################################
#############################################################################
"""
    Turns the list of lists of activations into binary vectors for each distinct kxk square evaluated in the list
"""

def activations_to_binary(activations:list):
    # Define the possible strings (c0 to c13)
    categories = [f'c{i}' for i in range(14)]
    
    # Map each category to its index
    category_to_index = {category: index for index, category in enumerate(categories)}
    
    # Convert each sublist into a binary vector
    binary_vectors = []
    for sublist in activations:
        # Create a binary vector initialized with 0s
        binary_vector = [0] * len(categories)
        # Set positions corresponding to the strings in the sublist to 1
        for string in sublist:
            if string in category_to_index:
                binary_vector[category_to_index[string]] = 1
        binary_vectors.append(binary_vector)
    
    return binary_vectors

#############################################################################
#############################################################################
"""
    Aggregates same type sensors in the same row, returning a flattened list of all sensors
"""

def sum_rows(bin:list, k:int, w:int, h:int):
    row_sums = []
    for row in range(h):
        row_sum = [0] * 14
        for col in range(w):
            row_sum += (bin[col + (h*row)])
        # print(row_sum)
        row_sums.append(row_sum.tolist())

    arr = np.array(row_sums)
    tot = arr.shape[0] * arr.shape[1]

    flat = arr.flatten()

    return flat, tot

#############################################################################
#############################################################################
"""
    MODIFIED FROM FUNCS.py: Completes a full step from image input to kmeans_amps update.
    
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

def new_kmeans_step(data:pd.DataFrame, image:int, kmeans_amps:list[int], k:int, 
         labels:pd.Series, kmeans_segmenter, n:int=784, num_sections:int=15):

    print(f"-----------------IMAGE {image}-----------------")

    real = labels[image]

    all_activs = all_kxk_activations(data=data, image=image, k=5)

    binarized = np.array(activations_to_binary(all_activs))

    row_sums, num_agg = sum_rows(binarized, k = 5, w = 24, h = 24)

    kmeans_output = row_sums + kmeans_amps
    # kmeans_output_binary = (kmeans_output > 5).astype(int)

    kmeans_section = kmeans_segmenter.assign_segment(kmeans_output)

    if kmeans_section == real:
        print(f"kmeans match: guess = {kmeans_section} , actual = {real}")
        adjustment = kmeans_segmenter.move_closer(vector=row_sums, closest=kmeans_section, alpha=.5)
        # print(adjustment)
        kmeans_amps[row_sums > 0] += adjustment[row_sums > 0]
        kmeans_correct = 1
    else: 
        print(f"kmeans no match: guess = {kmeans_section} , actual = {real}")
        kmeans_amps[row_sums == 0] = np.round(kmeans_amps[row_sums == 0] / 1, 1)
        kmeans_correct = 0

    return kmeans_amps, kmeans_correct

    
