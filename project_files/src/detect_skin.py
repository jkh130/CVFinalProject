import numpy as np

def detect_skin(image, positive_histogram, negative_histogram):
    """
    Detects skin in an image using positive and negative histograms.
    
    Parameters:
        image (numpy.ndarray): The input image array with shape (height, width, 3).
        positive_histogram (numpy.ndarray): The positive histogram.
        negative_histogram (numpy.ndarray): The negative histogram.
        
    Returns:
        numpy.ndarray: The output array with skin detection probabilities, shape (height, width).
    """
    
    histogram_bins = positive_histogram.shape[0]
    factor = 256 / histogram_bins
    
    # Calculate indices for each color channel
    red_indices = (image[:, :, 0] / factor).astype(int)
    green_indices = (image[:, :, 1] / factor).astype(int)
    blue_indices = (image[:, :, 2] / factor).astype(int)
    
    # Fetch probabilities from histograms using the indices
    skin_values = positive_histogram[red_indices, green_indices, blue_indices]
    non_skin_values = negative_histogram[red_indices, green_indices, blue_indices]
    
    # Compute total probabilities
    total = skin_values + non_skin_values
    
    # Calculate skin probabilities using Bayes rule: P(skin | RGB) = P(RGB | skin) * P(skin) / P(RGB)
    #      skin_vales = P(RGB | skin)
    # non_skin_values = P(RGB | non-skin)
    #  total = P(RGB) = P(RGB | skin) * P(skin) + P(RGB | non-skin) * P(non-skin). For simplicity, we assume P(skin) = P(non-skin) = 0.5
    #          result = P(skin | RGB)
    result = np.divide(skin_values, total, out=np.zeros_like(skin_values), where=total!=0)
    
    return result
