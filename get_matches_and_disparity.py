from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

import numpy as np
import matplotlib.pyplot as plt

def compute_disparity_map(image_shape, matching_points_left, matching_points_right, rootdir, CROP):
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
    if CROP:
        np.save(rootdir + "disparity_npy/" + obj + "_cropped.npy", disparity_map)
    else:
        np.save(rootdir + "disparity_npy/" + obj + ".npy", disparity_map)
    return disparity_map

if __name__ == '__main__':

    import os

    ### Change this to your own rgb pairs dir
    rootdir = '/home/lifanyu/Documents/ZED_data/'
    ### Change this for whether or not we want disparity for cropped images
    CROP = True
    RESIZE = True

    if CROP and RESIZE:
        spec = "_cropped_resized"
    elif CROP:
        spec = "_cropped"
    else:
        spec = ""

    folders = []

    image_pair_dir = rootdir + "rgb_pairs/"
    for entry in os.listdir(image_pair_dir):
        if os.path.isdir(os.path.join(image_pair_dir, entry)):
            folders.append(entry)

    for obj in folders:

        device = 'cuda'
        schedule = 'cosine'
        lr = 0.01
        niter = 300

        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        # you can put the path to a local checkpoint in model_name if needed
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

        images = load_images([rootdir + "rgb_pairs/" + obj + "/zed1" + spec + ".png", rootdir + "rgb_pairs/" + obj + "/zed2" + spec + ".png"], size = 512)
        
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=1,
                                                    device=device, dist='dot', 
                                                    #    block_size = 2 ** 14

                                                    block_size=2**13
                                                    )

        # ignore small border around the edge
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        # visualize a few matches
        import numpy as np
        import torch
        import torchvision.transforms.functional
        from matplotlib import pyplot as pl

        n_viz = 75
        num_matches = matches_im0.shape[0]
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        

        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

        viz_imgs = []
        for i, view in enumerate([view1, view2]):
            rgb_tensor = view['img'] * image_std + image_mean
            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        pl.figure(figsize=(30, 20))
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)

        if CROP:
            plt.savefig(rootdir + "disparity/" + obj + "_cropped_matches.png")
        else:
            plt.savefig(rootdir + "disparity/" + obj + "_matches.png")
        pl.show(block=True)

        ################################# disparity
        print("\nmatches shape", matches_im0.shape)
        image_shape = (H0, W0)

        # Example matching points
        matching_points_left = matches_im0
        # viz_matches_im0  # Points in the left image
        matching_points_right = matches_im1
        # viz_matches_im1  # Corresponding points in the right image

        # Compute disparity map
        disparity_map = compute_disparity_map(image_shape, matching_points_left, matching_points_right, rootdir, CROP)

        # Normalize disparity map for better visualization
        # disparity_map_normalized = (disparity_map - np.min(disparity_map)) / (np.max(disparity_map) - np.min(disparity_map) + 1e-5)
        # disparity_map_normalized = (disparity_map_normalized * 255).astype(np.uint8)

        # Plot the disparity map
        plt.figure(figsize=(30, 20))

        # Change this if needed: remove the vmin and vmax to see the most common range for the disparity values
        # and add vmin, vmax back for better visualizations and comparisons
        plt.imshow(disparity_map, cmap='jet', vmin=-25, vmax=175)
        plt.colorbar(label='Disparity')
        plt.title('Disparity Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        if CROP:
            plt.savefig(rootdir + "disparity/" + obj + "_cropped.png") 
        else:
            plt.savefig(rootdir + "disparity/" + obj + ".png") 
        plt.show()
