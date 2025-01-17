import cv2
import numpy as np
import matplotlib.pyplot as plt

# def joint_bilateral_filter(disparity, guidance, diameter=5, sigma_s=10, sigma_r=0.1):
#     """
#     Apply joint bilateral filter to densify a disparity map.

#     Args:
#         disparity (ndarray): Sparse or noisy disparity map (grayscale or single-channel float).
#         guidance (ndarray): Guidance image (RGB or grayscale).
#         diameter (int): Diameter of the kernel.
#         sigma_s (float): Spatial standard deviation.
#         sigma_r (float): Intensity range standard deviation.

#     Returns:
#         ndarray: Densified disparity map.
#     """
#     # Ensure disparity is single-channel
#     if len(disparity.shape) == 3:
#         raise ValueError("Disparity map must be single-channel.")
    
#     # Apply joint bilateral filter
#     filtered_disparity = cv2.ximgproc.jointBilateralFilter(
#         guidance, disparity, d=diameter, sigmaColor=sigma_r, sigmaSpace=sigma_s
#     )
    
#     return filtered_disparity


import cv2
import numpy as np

def joint_bilateral_filter(disparity, guidance, diameter=5, sigma_s=10, sigma_r=0.1):
    """
    Apply joint bilateral filter to densify a disparity map.

    Args:
        disparity (ndarray): Sparse disparity map (grayscale, single-channel, CV_8U or CV_32F).
        guidance (ndarray): Guidance image (RGB or grayscale, same depth as disparity).
        diameter (int): Diameter of the kernel.
        sigma_s (float): Spatial standard deviation.
        sigma_r (float): Intensity range standard deviation.

    Returns:
        ndarray: Densified disparity map.
    """
    # Ensure disparity is single-channel
    if len(disparity.shape) == 3:
        raise ValueError("Disparity map must be single-channel.")
    
    # Convert both disparity and guidance to float32
    if disparity.dtype != np.float32:
        disparity = disparity.astype(np.float32)
    if guidance.dtype != np.float32:
        guidance = guidance.astype(np.float32)
    
    # Apply joint bilateral filter
    filtered_disparity = cv2.ximgproc.jointBilateralFilter(
        guidance, disparity, d=diameter, sigmaColor=sigma_r, sigmaSpace=sigma_s
    )
    return filtered_disparity



import numpy as np
import cv2

def joint_bilateral_filter_numpy(disparity, guidance, diameter=15, sigma_s=50, sigma_r=0.5):
    """
    Apply a manual joint bilateral filter using NumPy.

    Args:
        disparity (ndarray): Sparse disparity map (single-channel, float32).
        guidance (ndarray): Guidance image (RGB or grayscale, same depth as disparity).
        diameter (int): Diameter of the kernel (kernel size).
        sigma_s (float): Spatial standard deviation.
        sigma_r (float): Intensity range standard deviation.

    Returns:
        ndarray: Densified disparity map.
    """
    # Get the height and width of the disparity map
    h, w = disparity.shape

    # Calculate the radius of the kernel
    radius = diameter // 2

    # Initialize an empty output image
    filtered_disparity = np.zeros_like(disparity)

    # Pad the disparity and guidance images to handle borders
    disparity_padded = np.pad(disparity, ((radius, radius), (radius, radius)), mode='constant', constant_values=0)
    guidance_padded = np.pad(guidance, ((radius, radius), (radius, radius), (0, 0)), mode='constant')

    # Gaussian spatial kernel
    spatial_kernel = np.exp(-0.5 * (np.arange(-radius, radius+1)**2) / (sigma_s**2))
    spatial_kernel = np.outer(spatial_kernel, spatial_kernel)  # 2D spatial kernel

    # Loop over every pixel in the disparity map
    for y in range(h):
        for x in range(w):
            # Extract local window of disparity and guidance
            disparity_window = disparity_padded[y:y+diameter, x:x+diameter]
            guidance_window = guidance_padded[y:y+diameter, x:x+diameter]

            # Compute the range kernel based on the difference between the guidance window and the current pixel
            intensity_diff = np.linalg.norm(guidance_window - guidance_padded[y+radius, x+radius], axis=2)
            range_kernel = np.exp(-0.5 * (intensity_diff**2) / (sigma_r**2))

            # Combine spatial and range kernels
            bilateral_kernel = spatial_kernel * range_kernel

            # Apply the filter to the disparity window
            weight_sum = np.sum(bilateral_kernel)
            filtered_disparity[y, x] = np.sum(bilateral_kernel * disparity_window) / weight_sum if weight_sum != 0 else 0

    return filtered_disparity




if __name__ == '__main__':

    ### Change this
    rootdir = '/home/lifanyu/Downloads/ZED+mast3r/'
    obj = "paint_front"

    disparity_npy = rootdir + "disparity_npy/" + obj + ".npy"
    rgb_img = rootdir + "rgb_pairs/" + obj + "/zed2.png"

    ######################## CV@ ########################

    # # Load the disparity map and guidance image
    # disparity = np.load(disparity_npy)  # Disparity map
    # guidance = cv2.imread(rgb_img, cv2.IMREAD_COLOR)  # Guidance image

    # # Resize the guidance image to match the disparity map dimensions
    # height, width = disparity.shape
    # guidance_resized = cv2.resize(guidance, (width, height), interpolation=cv2.INTER_LINEAR)

    # # Apply the joint bilateral filter
    # densified_disparity = joint_bilateral_filter(disparity, guidance_resized)

    # # Save the results
    # np.save('densified_disparity.npy', densified_disparity)
    # cv2.imwrite('densified_disparity.png', (densified_disparity * 255).astype(np.uint8))

    ############################ numpy ##########################
    # Example usage
    disparity = np.load(disparity_npy)  # Load disparity map
    guidance = cv2.imread(rgb_img, cv2.IMREAD_COLOR)  # Load guidance image

    # Modify these parameters for denser output
    diameter = 15  # Larger diameter
    sigma_s = 50   # Increase spatial standard deviation
    sigma_r = 0.5  # Increase intensity standard deviation

    height, width = disparity.shape
    guidance_resized = cv2.resize(guidance, (width, height), interpolation=cv2.INTER_LINEAR)

    # Apply the joint bilateral filter using NumPy
    densified_disparity = joint_bilateral_filter_numpy(disparity, guidance_resized)

    # Apply the filter multiple times for a denser output
    for _ in range(10):
        densified_disparity = joint_bilateral_filter_numpy(densified_disparity, guidance, diameter=diameter, sigma_s=sigma_s, sigma_r=sigma_r)

    # Save the result
    np.save('densified_disparity.npy', densified_disparity)
    cv2.imwrite('densified_disparity.png', (densified_disparity * 255).astype(np.uint8))

