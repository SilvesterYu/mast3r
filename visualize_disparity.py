import matplotlib.pyplot as plt
import numpy as np

def visualize_disparity_map(disparity_map):
    # Plot the disparity map
    plt.figure(figsize=(30, 20))
    plt.imshow(disparity_map, cmap='jet', vmin=-25, vmax=175)
    plt.colorbar(label='Disparity')
    plt.title('Disparity Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig("disparity.png") 
    plt.show()

visualize_disparity_map(np.load("densified_disparity.npy"))