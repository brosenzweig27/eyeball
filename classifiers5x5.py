import pandas as pd
import numpy as np

#############################################################################
#############################################################################
"""
    Classifier type 0: empty
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 0, False otherwise.

"""

def d0(signals: list):
    # Unpack relevant values from the assessed square
    is_interesting = signals[-1]
    avg_value = signals[0]

    # Empty condition check
    if (not is_interesting):
        if avg_value < 100:
            return True
        else:
            return False
    else: 
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 1: full
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 1, False otherwise.

"""

def d1(signals: list):
    # Unpack relevant values from the assessed square
    is_interesting = signals[-1]
    avg_value = signals[0]

    # Full condition check
    if (not is_interesting):
        if avg_value >= 100:
            return True
        else:
            return False
    else: 
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 2: Dark vbar on the left
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 2, False otherwise.

"""

def d2(values_list: list[int], signals: list, type: str):
    # Speed things up by first checking if it's intersting:
    # If not, check if full or empty
    is_interesting = signals[5]
    avg = signals[0]

    if (not is_interesting):
        if type == 'empty':
            if avg < 100:
                return True
            else:
                return False
        elif type == 'full':
            if avg >= 100:
                return True
            else:
                return False
        else:
            return False
    else:

        # Unpack rest of relevant values from assess()
        dark_cols = signals[1]
        dark_rows = signals[2]
        light_cols = signals[3]
        light_rows = signals[4]

        # Define the binary masks for "dark" and "light"
        dark_threshold = avg + 40
        light_threshold = avg - 40
        dark_mask = values_list >= dark_threshold
        light_mask = values_list <= light_threshold

        # Define dark left, middle, and right
        dark_left = (np.all(dark_mask[:,0]) or (np.all(dark_mask[:,0]) and np.all(dark_mask[:,1])))
        dark_middle_col = (
            np.any(dark_mask[:, 1]) or  # Column 2
            np.any(dark_mask[:, 2]) or  # Column 3
            np.any(dark_mask[:, 3]) or  # Column 4
            np.all(dark_mask[:, 1:2], axis=1) or  # Columns 2 and 3
            np.all(dark_mask[:, 2:3], axis=1) or  # Columns 3 and 4
            np.all(dark_mask[:, 1,3], axis=1)  # All three columns
        )
        dark_right = (np.all(dark_mask[:,4]) or (np.all(dark_mask[:,3]) and np.all(dark_mask[:,4])))

        # Define dark top, middle, and bottom
        dark_top = (np.all(dark_mask[0,:]) or (np.all(dark_mask[0,:]) and np.all(dark_mask[1,:])))
        dark_middle_row = (
            np.any(dark_mask[1,:]) or  # Row 2
            np.any(dark_mask[2,:]) or  # Row 3
            np.any(dark_mask[3,:]) or  # Row 4
            np.all(dark_mask[1:2,:], axis=1) or  # Rows 2 and 3
            np.all(dark_mask[2:3,:], axis=1) or  # Rows 3 and 4
            np.all(dark_mask[1,3,:], axis=1)  # All three rows
        )
        dark_bottom = (np.all(dark_mask[4,:]) or (np.all(dark_mask[3,:]) and np.all(dark_mask[4,:])))

        # Check sensor type and return relevant truth value

        #
    
