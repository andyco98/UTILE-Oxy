import numpy as np
from patchify import patchify, unpatchify
from PIL import Image
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from keras.models import load_model

def prediction(image_path, model_path, mask_folder):
    patch_size = 256
    n_classes = 1

    BACKBONE1 = 'resnext101'
    preprocess_input1 = sm.get_preprocessing(BACKBONE1)

    model1 = load_model(model_path, compile=False)

    test_list = os.listdir(image_path)

    for name in test_list:
        patch_list = []  
        img = Image.open(image_path+name).convert("L").resize((512, 512))
        array = np.array(img)
        print(array.shape)
        patches = patchify(array, (256,256), step=256)
        print(patches.shape)
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i,j,:,:]
                single_patch = np.stack((single_patch,)*3, axis=-1)
                test_img_input=np.expand_dims(single_patch, 0)
                #test_img_input = preprocess_input1(test_img_input)
                test_prediction1 = (model1.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
                patch_list.append(test_prediction1)

        test_img = np.array(patch_list)
        predicted_patches_reshaped = np.reshape(test_img, (patches.shape[0], patches.shape[1], 256, 256) )
        print(predicted_patches_reshaped.shape)
        recon_img = unpatchify(predicted_patches_reshaped, (512,512))
        recon_img = Image.fromarray(np.uint8(recon_img*255)).save(mask_folder + f"/pred_{name}.png")

    return