import numpy as np
from PIL import Image
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from keras.models import load_model
import imageio
import re


def extract_patches(image, patch_size=256, overlap=128):
    patches = []
    coords = []
    height, width = image.shape

    step = patch_size - overlap

    for y in range(0, height - overlap, step):
        for x in range(0, width - overlap, step):
            y_end = y + patch_size
            x_end = x + patch_size

            if y_end > height:
                y_start = height - patch_size
                y_end = height
            else:
                y_start = y

            if x_end > width:
                x_start = width - patch_size
                x_end = width
            else:
                x_start = x

            patch = image[y_start:y_end, x_start:x_end]
            patches.append(patch)
            coords.append((y_start, x_start))
    return patches, coords

def create_gaussian_weight(patch_size, sigma):
    center = patch_size // 2
    ax = np.linspace(-center, center, patch_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.max()
    return kernel

def rebuild_image(patches, coords, image_shape, patch_size=256, overlap=128, sigma=32):
    height, width = image_shape
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    weight_map = np.zeros(image_shape, dtype=np.float32)
    
    gaussian_weight = create_gaussian_weight(patch_size, sigma)

    for patch, (y, x) in zip(patches, coords):
        y_end = min(y + patch_size, height)
        x_end = min(x + patch_size, width)

        patch_weight = gaussian_weight[:y_end - y, :x_end - x]

        reconstructed[y:y_end, x:x_end] += patch * patch_weight
        weight_map[y:y_end, x:x_end] += patch_weight

    reconstructed /= weight_map
    return reconstructed

def process_and_predict_image(image, model, patch_size=256, overlap=128, sigma=32):
    # Pad the image if necessary
    height, width = image.shape
    pad_height = (patch_size - (height - overlap) % (patch_size - overlap)) % (patch_size - overlap)
    pad_width = (patch_size - (width - overlap) % (patch_size - overlap)) % (patch_size - overlap)

    image_padded = np.pad(image, ((0, pad_height), (0, pad_width)), mode='reflect')
    image_shape = image_padded.shape

    # Extract patches
    patches, coords = extract_patches(image_padded, patch_size, overlap)

    # Predict on each patch
    predicted_patches = []
    counter = 1
    total_patches = len(patches)
    for patch in patches:
        # Convert grayscale to RGB by stacking the channels
        #if len(patch.shape) == 2:
        patch = np.stack((patch,)*3, axis=-1)
        #patch = patch.astype(np.float32) / 255.0
        patch = np.expand_dims(patch, axis=0)  # Add batch dimension

        # If preprocessing is required by your model, uncomment the next line
        # patch = preprocess_input1(patch)
        prediction = (model.predict(patch)[0,:,:,0] > 0.5).astype(np.uint8)
        #prediction = model.predict(patch)
        predicted_patches.append(prediction)
        print(f'Patch processed: {counter} of {total_patches}')
        counter += 1

    # Rebuild the image
    predicted_image = rebuild_image(predicted_patches, coords, image_shape, patch_size, overlap, sigma)
    
    # Crop to original size
    predicted_image = predicted_image[:height, :width]
    
    # Threshold the predicted image for binary segmentation
    predicted_image = (predicted_image > 0.5).astype(np.uint8)
    
    return predicted_image

def prediction_patch(image_path, model_path, mask_folder, resize = True):
    patch_size = 256
    n_classes = 1

    BACKBONE1 = 'resnext101'
    preprocess_input1 = sm.get_preprocessing(BACKBONE1)
    patch_size = 256
    overlap = 128
    sigma = 32
    model1 = load_model(model_path, compile=False)

    test_list = os.listdir(image_path)

    for name in test_list:
        img_path = os.path.join(image_path, name)
        img = Image.open(img_path).convert("L")
        w,h = img.size
        img = img.resize((512, 512))
        array = np.array(img)
        print(f"Processing image: {name}, shape: {array.shape}")
    
        predicted_image = process_and_predict_image(array, model1, patch_size=patch_size, overlap=overlap, sigma=sigma)
        print("Patch prediction unique values:", np.unique(predicted_image))

        if resize:
            recon_img = Image.fromarray(np.uint8(predicted_image*255)).resize((w,h), resample=Image.NEAREST).save(mask_folder + f"/pred_{name}.png")
        else: recon_img = Image.fromarray(np.uint8(predicted_image*255)).save(mask_folder + f"/pred_{name}.png")

    return


def prediction_nopatch(image_path, model_path, mask_folder, resize = True):

    BACKBONE1 = 'resnext101'
    preprocess_input1 = sm.get_preprocessing(BACKBONE1)

    model1 = load_model(model_path, compile=False)

    test_list = os.listdir(image_path)

    for name in test_list:
        patch_list = []  
        img = Image.open(image_path+name).convert("L")
        w,h = img.size
        img = img.resize((512,512))
        array = np.array(img)
        print(array.shape)
        single_patch = np.stack((array,)*3, axis=-1)
        test_img_input=np.expand_dims(single_patch, 0)
        #test_img_input = preprocess_input1(test_img_input)
        test_prediction1 = (model1.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
        if resize:
            im = Image.fromarray(test_prediction1*255).resize((w,h), resample=Image.NEAREST)
        else: im = Image.fromarray(test_prediction1*255)
        im.save(mask_folder + f"/pred_{name}.png")
    return


def numerical_sort(value):
    """
    Helper function to extract numbers from the filename for sorting.
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def gif_creation(input_folder, output_filename, frame_duration):
    images = []
    images_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.lower().endswith(('.tif','.tiff','.png', '.jpg', '.jpeg', '.bmp'))]
    images_paths.sort(key=numerical_sort)  # Sort files by name

    for filename in images_paths:
        file_path = os.path.join(input_folder, filename)
        images.append(imageio.imread(filename))

    imageio.mimsave(output_filename, images, duration=frame_duration)