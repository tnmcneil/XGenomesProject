import time
import os
import os.path
from os.path import join as path_join
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import numpy as np
import tensorflow as tf
from keras_retinanet.utils.anchors import AnchorParameters
from PIL import Image


# Here are some notes for our settings on backbone network and anchor-box parameters.

# Variables for loading model and output
backbone_name = "resnet50"

# Anchor parmaeters used for training DNA detection
# Since we customize these parameters, it will cause error when directly applying modle
# with default parameters
anchor_params = AnchorParameters(
    [16, 32, 64, 128, 256],
    [8, 16, 32, 64, 128],
    [7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 40, 50],
    [0.75, 1, 1.2, 1.4, 1.6]
)

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

# Start TensorFlow session
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

class DNA_Cropper():
    def __init__(self, model_path, backbone, anchor_params):
        self._model_path = model_path
        self._backbone = backbone
        self._anchor_params = anchor_params

        keras.backend.tensorflow_backend.set_session(get_session())
        self._pred_model = models.load_model(self._model_path, backbone_name=self._backbone)
        self._pred_model = models.convert_model(self._pred_model, anchor_params=self._anchor_params)

        # By default our model should not detect anything of Else category
        self._labels_to_names = {0: 'DNA-Sequence', 1: 'Else'}

    def crop_pictures(self, filenames, out_paths, out_tags, threshold=0.5):
        """
            Crop detected DNA strands out of image.

            Params:
                filenames: list of strings indicate input image files.
                out_paths: list of strings indicate output directories correspond to input images
                out_tags: list of strings indicate tags used in output images correspond to input images

            Return:
                No return value (None)
        """

        assert len(filenames) == len(out_paths) == len(out_tags)

        for filename, out_path, out_tag in zip(filenames, out_paths, out_tags):
            image = read_image_bgr(filename)
            image_out = image.copy()

            # Resize input image and use our model to detect objects
            print("Detecting file {}...".format(filename))
            image = preprocess_image(image)
            image, scale = resize_image(image)
            time_start = time.time()
            boxes, scores, labels = self._pred_model.predict_on_batch(np.expand_dims(image, axis=0))
            print("processing time: ", time.time() - time_start)
            # Correct scale.
            boxes /= scale

            makedirs(out_path)
            # win_index
            win_idx = 0
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
                if score > threshold:
                    # Crop image and save to another file
                    win_idx += 1
                    crop_out_filename = "{}_piece_{:0>4d}.jpg".format(out_tag, win_idx)
                    # print("Writing out to {}...".format(crop_out_filename))
                    crop_out_path = path_join(out_path, crop_out_filename)
                    b = box.astype(int)
                    cropped_obj = image_out[b[1]:b[3]+1, b[0]:b[2]+1, :]
                    image_pil = Image.fromarray(cropped_obj)
                    image_pil.save(crop_out_path)
        print("done!")

# Example
trained_model_path = "./snapshots/resnet50_csv_40.h5"
input_pics = ["./InputPics/lambda_1-superRes-output.png"]
output_paths = ["./CropOutput/"]
output_tag = ["testcropmodule"]

cropper = DNA_Cropper(trained_model_path, backbone_name, anchor_params)
cropper.crop_pictures(input_pics, output_paths, output_tag, 0.45)
