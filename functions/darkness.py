import numpy as np

def bars(signals:list, values_list: list[int]):
    k = len(values_list)
    nleft = np.floor(k/2).astype(int)
    nright = np.ceil(k/2).astype(int)

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

    if (not is_interesting):
        if avg < 100:
            classifications.append('c0')
        else:
            classifications.append('c1')
    
    else:
        if (dark_cols or light_cols):
            if (
                any(np.all(dark_mask[:, :n], axis=0) for n in range(nleft))             # Dark left
                or 
                any(np.all(light_mask[:, 1+n:], axis=0) for n in range(nleft))          # Light middle and right
            ):
                classifications.append('c2')
            
            elif (
                any(np.all(light_mask[:, -(n+1):], axis=0) for n in range(nright))      # Light right
                or
                any(np.all(dark_mask[:, :nleft+n], axis=0) for n in range(nright-1))    # Dark left and middle
            ):
                classifications.append('c3')
            
            elif (
                (
                    any(np.all(dark_mask[:, :n], axis=0) for n in range(nleft))         # Dark left
                    and
                    any(np.all(dark_mask[:, -(n+1):], axis=0) for n in range(nright))   # Dark right
                )
                or
                any(np.any(np.all(light_mask[:, n+1:m+1], axis=0)) for n in range(k-1) for m in range(n+1, k))    # Light middle
            ):
                classifications.append('c4')