import os
from PIL import Image

def setup_directories(mask_folder):
    """
    Sets up the parent directory based on the mask_folder path.
    
    Parameters:
    - mask_folder (str): Path to the mask folder.
    
    Returns:
    - parent_dir (str): Parent directory of the mask folder.
    """
    global parent_dir_of_mask_folder  # Declare it as global for use in subsequent cells
    parent_dir_of_mask_folder = os.path.dirname(mask_folder.rstrip('/'))  # Ensure no trailing slash
    return parent_dir_of_mask_folder

def create_binary_tiff_stack(input_dir, output_dir, output_filename, chunk_size=500):
    """
    Creates a TIFF stack from PNG images from masks for tracking purposes.
    
    Parameters:
    - input_dir (str): this is the mask_folder from previous cells
    - output_dir (str): Directory to save the output TIFF stack.
    - output_filename (str): Name of the output TIFF file.
    - chunk_size (int): Number of images to process at once (500 works reasonably well).
    
    """
    
    # Check if the input directory exists
    if not os.path.exists(input_dir):
        return f"Input directory does not exist: {input_dir}"

    # Create a list of all PNG files in the input directory
    png_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    # Check if there are any PNG files in the directory
    if len(png_files) == 0:
        return f"No PNG files found in the directory: {input_dir}"

    # Sort the files to ensure they are in the correct order
    png_files.sort()

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the output TIFF file path
    output_path = os.path.join(output_dir, output_filename)
    
    # Check if the file already exists and remove it if necessary
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Process images in chunks
    for i in range(0, len(png_files), chunk_size):
        chunk_files = png_files[i:i + chunk_size]
        images = []

        for file in chunk_files:
            img_path = os.path.join(input_dir, file)
            img = Image.open(img_path)
            images.append(img)
        
        # Append or create the TIFF file
        if i == 0:
            # First chunk - create the TIFF file
            images[0].save(output_path, save_all=True, append_images=images[1:], compression="tiff_deflate")
        else:
            # Subsequent chunks - append to the existing TIFF file
            images[0].save(output_path, save_all=True, append_images=images[1:], compression="tiff_deflate", append=True)

        print(f"Processed chunk {i//chunk_size + 1}/{(len(png_files) + chunk_size - 1)//chunk_size}")

    return f"TIFF stack saved at: {output_path}"

