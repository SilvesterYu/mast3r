from PIL import Image
import os

def crop_png(image_path, output_path, x_start, y_start, width, height):
    """
    Crop a rectangular region from a PNG image.

    Parameters:
        image_path: Path to the input PNG image.
        output_path: Path to save the cropped PNG image.
        x_start: X-coordinate of the top-left corner of the cropping rectangle.
        y_start: Y-coordinate of the top-left corner of the cropping rectangle.
        width: Width of the cropping rectangle.
        height: Height of the cropping rectangle.

    Returns:
        Saves the cropped image to the output path.
    """
    # Open the PNG image
    image = Image.open(image_path)
    w, h = image.size
    x_start = int(0.25 * w)
    y_start = int(0.25 * h)
    W = int(0.5 * w)
    H = int(0.5 * h)

    print("H", "W", H, W)

    # Define the cropping rectangle
    crop_box = (x_start, y_start, x_start + W, y_start + H)

    # Crop the image
    cropped_image = image.crop(crop_box)

    # Save the cropped image
    cropped_image.save(output_path, "PNG")

# Example usage
if __name__ == "__main__":

    rootdir = '/home/lifanyu/Downloads/ZED+mast3r/rgb_pairs/'
    folders = []
    for entry in os.listdir(rootdir):
        if os.path.isdir(os.path.join(rootdir, entry)):
            folders.append(entry)
    
    for obj in folders:
    
        input_path = rootdir + obj + "/zed1.png"
        output_path = rootdir + obj + "/zed1_cropped.png"
        x_start, y_start = 100, 50
        width, height = 200, 150

        crop_png(input_path, output_path, x_start, y_start, width, height)
        print(f"Cropped image saved to {output_path}")

        input_path = rootdir + obj + "/zed2.png"
        output_path = rootdir + obj + "/zed2_cropped.png"
        x_start, y_start = 100, 50
        width, height = 200, 150

        crop_png(input_path, output_path, x_start, y_start, width, height)
        print(f"Cropped image saved to {output_path}")