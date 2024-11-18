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

def c0(signals: list):
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

def c1(signals: list):
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

def c2(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_cols = signals[1]
    dark_rows = signals[2]
    light_cols = signals[3]
    light_rows = signals[4]
    is_important = signals[5]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light columns
    if (not (dark_cols or light_cols)):
        return False
    # Dark left, not dark middle or right
    elif (np.all(dark_mask[:,0]) and (not np.all(dark_mask[:,1])) and (not np.all(dark_mask[:,2]))):
        return True
    # Not light left, light middle and light right
    elif ((not np.all(light_mask[:,0])) and np.all(light_mask[:,1]) and np.all(light_mask[:,2])):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 3: Light vbar on the right
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 3, False otherwise.

"""

def c3(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_cols = signals[1]
    light_cols = signals[3]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light columns
    if (not (dark_cols or light_cols)):
        return False
    # Dark left and middle, no dark right
    elif ((np.all(dark_mask[:,0])) and (np.all(dark_mask[:,1])) and (not np.all(dark_mask[:,2]))):
        return True
    # No light left of middle, light right
    elif ((not np.all(light_mask[:,0])) and (not np.all(light_mask[:,1])) and (np.all(light_mask[:,2]))):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 4: Light vbar in the middle
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 4, False otherwise.

"""

def c4(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_cols = signals[1]
    light_cols = signals[3]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light columns
    if (not (dark_cols or light_cols)):
        return False
    # Dark left and right, no dark middle
    elif ((np.all(dark_mask[:,0])) and (not np.all(dark_mask[:,1])) and (np.all(dark_mask[:,2]))):
        return True
    # No light left of right, light middle
    elif ((not np.all(light_mask[:,0])) and (np.all(light_mask[:,1])) and (not np.all(light_mask[:,2]))):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 5: Dark vbar in the middle.
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 5, False otherwise.

"""

def c5(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_cols = signals[1]
    light_cols = signals[3]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light columns
    if (not (dark_cols or light_cols)):
        return False
    # No dark left or right, dark middle
    elif ((not np.all(dark_mask[:,0])) and (np.all(dark_mask[:,1])) and (not np.all(dark_mask[:,2]))):
        return True
    # Light left and right, no light middle
    elif ((np.all(light_mask[:,0])) and (not np.all(light_mask[:,1])) and (np.all(light_mask[:,2]))):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 6: Light vbar on the left.
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 6, False otherwise.

"""

def c6(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_cols = signals[1]
    light_cols = signals[3]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light columns
    if (not (dark_cols or light_cols)):
        return False
    # No dark left, dark middle and right
    elif ((not np.all(dark_mask[:,0])) and (np.all(dark_mask[:,1])) and (np.all(dark_mask[:,2]))):
        return True
    # Light left, not light middle and right
    elif ((np.all(light_mask[:,0])) and (not np.all(light_mask[:,1])) and (not np.all(light_mask[:,2]))):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 7: Dark vbar on the right.
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 7, False otherwise.

"""

def c7(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_cols = signals[1]
    light_cols = signals[3]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light columns
    if (not (dark_cols or light_cols)):
        return False
    # No dark left or middle, dark right
    elif ((not np.all(dark_mask[:,0])) and (not np.all(dark_mask[:,1])) and (np.all(dark_mask[:,2]))):
        return True
    # Light left and middle, no light right
    elif ((np.all(light_mask[:,0])) and (np.all(light_mask[:,1])) and (not np.all(light_mask[:,2]))):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 8: Dark hbar on top.
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 8, False otherwise.

"""

def c8(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_rows = signals[2]
    light_rows = signals[4]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light rows
    if (not (dark_rows or light_rows)):
        return False
    # Dark top, no dark middle or bottom
    elif ((np.all(dark_mask[0,:])) and (not np.all(dark_mask[1,:])) and (not np.all(dark_mask[2,:]))):
        return True
    # No light top, light middle and bottom
    elif ((not np.all(light_mask[0,:])) and (np.all(light_mask[1,:])) and (np.all(light_mask[2,:]))):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 9: Light hbar on the bottom.
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 9, False otherwise.

"""

def c9(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_rows = signals[2]
    light_rows = signals[4]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light rows
    if (not (dark_rows or light_rows)):
        return False
    # Dark on top and middle, no dark on bottom
    elif ((np.all(dark_mask[0,:])) and (np.all(dark_mask[1,:])) and (not np.all(dark_mask[2,:]))):
        return True
    # No light top or middle, light on bottom
    elif ((not np.all(light_mask[0,:])) and (not np.all(light_mask[1,:])) and (np.all(light_mask[2,:]))):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 10: Light hbar in the middle.
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 10, False otherwise.

"""

def c10(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_rows = signals[2]
    light_rows = signals[4]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light rows
    if (not (dark_rows or light_rows)):
        return False
    # Dark on top and bottom, no dark in middle
    elif ((np.all(dark_mask[0,:])) and (not np.all(dark_mask[1,:])) and (np.all(dark_mask[2,:]))):
        return True
    # No light on top or bottom, light in middle
    elif ((not np.all(light_mask[0,:])) and (np.all(light_mask[1,:])) and (not np.all(light_mask[2,:]))):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 11: Dark hbar in the middle.
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 11, False otherwise.

"""

def c11(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_rows = signals[2]
    light_rows = signals[4]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light rows
    if (not (dark_rows or light_rows)):
        return False
    # No dark on top or bottom, dark in the middle
    elif ((not np.all(dark_mask[0,:])) and (np.all(dark_mask[1,:])) and (not np.all(dark_mask[2,:]))):
        return True
    # Light on top and bottom, not light in the middle
    elif ((np.all(light_mask[0,:])) and (not np.all(light_mask[1,:])) and (np.all(light_mask[2,:]))):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 12: Light hbar on the top.
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 12, False otherwise.

"""

def c12(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_rows = signals[2]
    light_rows = signals[4]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light rows
    if (not (dark_rows or light_rows)):
        return False
    # No dark on top, dark in middle and bottom
    elif ((not np.all(dark_mask[0,:])) and (np.all(dark_mask[1,:])) and (np.all(dark_mask[2,:]))):
        return True
    # Light on top, not light in the middle or bottom
    elif ((np.all(light_mask[0,:])) and (not np.all(light_mask[1,:])) and (not np.all(light_mask[2,:]))):
        return True
    # All other cases
    else:
        return False
    
#############################################################################
#############################################################################
"""
    Classifier type 13: Dark hbar on the bottom.
    
    Parameters:
    - outputs of assess(): List[bool], length 6.

    Returns:
    - bool: True if classified into group 13, False otherwise.

"""

def c13(values_list: list[int], signals: list):
    # Unpack relevant values from assess()
    avg = signals[0]
    dark_rows = signals[2]
    light_rows = signals[4]

    # Define the binary masks for "dark" and "light"
    dark_threshold = avg + 40
    light_threshold = avg - 40
    dark_mask = values_list >= dark_threshold
    light_mask = values_list <= light_threshold

    # No dark or light rows
    if (not (dark_rows or light_rows)):
        return False
    # Not dark on top or middle, dark on the bottom
    elif ((not np.all(dark_mask[0,:])) and (not np.all(dark_mask[1,:])) and (np.all(dark_mask[2,:]))):
        return True
    # Light on the top and middle, not light on the bottom
    elif ((np.all(light_mask[0,:])) and (np.all(light_mask[1,:])) and (not np.all(light_mask[2,:]))):
        return True
    # All other cases
    else:
        return False