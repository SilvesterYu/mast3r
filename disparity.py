import numpy as np
import matplotlib.pyplot as plt

def compute_disparity_map(image_shape, matching_points_left, matching_points_right):
    """
    Compute a disparity map from matching points.

    Parameters:
        image_shape: Tuple (height, width) of the image.
        matching_points_left: List of (x, y) coordinates in the left image.
        matching_points_right: List of (x, y) coordinates in the right image.

    Returns:
        disparity_map: The computed disparity map as a 2D array.
    """
    h, w = image_shape
    disparity_map = np.zeros((h, w), dtype=np.float32)

    # Compute disparities for each matching point pair
    for (x_left, y_left), (x_right, y_right) in zip(matching_points_left, matching_points_right):
        x_left, y_left, x_right, y_right = map(int, (x_left, y_left, x_right, y_right))
        if 0 <= x_left < w and 0 <= y_left < h and 0 <= x_right < w and 0 <= y_right < h:
            disparity = x_left - x_right  # Compute horizontal disparity
            disparity_map[y_left, x_left] = disparity

    return disparity_map

# Example Usage
if __name__ == "__main__":
    # Image shape (height, width)
    image_shape = (400, 600)

    # Example matching points
    matching_points_left = [(150, 200), (300, 250), (400, 300), (500, 350)]  # Points in the left image
    matching_points_right = [(140, 200), (290, 250), (390, 300), (480, 350)]  # Corresponding points in the right image

    # Compute disparity map
    disparity_map = compute_disparity_map(image_shape, matching_points_left, matching_points_right)

    # Normalize disparity map for better visualization
    disparity_map_normalized = (disparity_map - np.min(disparity_map)) / (np.max(disparity_map) - np.min(disparity_map) + 1e-5)
    disparity_map_normalized = (disparity_map_normalized * 255).astype(np.uint8)

    # Plot the disparity map
    plt.figure(figsize=(10, 5))
    plt.imshow(disparity_map_normalized, cmap='jet')
    plt.colorbar(label='Disparity')
    plt.title('Disparity Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
