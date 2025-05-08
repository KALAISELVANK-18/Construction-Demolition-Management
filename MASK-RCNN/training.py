import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw
import tensorflow.compat.v1 as tf
import skimage.draw
from matplotlib import pyplot as plt

from mrcnn import utils, model as modellib
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes


class CocoLikeDataset(utils.Dataset):
    def load_data(self, annotation_json, images_dir):
        with open(annotation_json) as json_file:
            coco_json = json.load(json_file)

        source_name = "coco_like"
        for category in coco_json['categories']:
            if category['id'] > 0:
                self.add_class(source_name, category['id'], category['name'])

        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            annotations.setdefault(image_id, []).append(annotation)

        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                continue
            seen_images[image_id] = image

            try:
                image_path = os.path.abspath(os.path.join(images_dir, image['file_name']))
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image['width'],
                    height=image['height'],
                    annotations=annotations[image_id]
                )
            except KeyError as key:
                print(f"Skipping image {image_id} due to missing key: {key}")

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')

            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)

            instance_masks.append(np.array(mask) > 0)
            class_ids.append(class_id)

        if instance_masks:
            mask = np.stack(instance_masks, axis=-1)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return np.empty((0, 0, 0)), np.array([])


class CDConfig(Config):
    NAME = "C&D_waste_management"
    NUM_CLASSES = 1 + 3  # background + 3 classes
    STEPS_PER_EPOCH = 100
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640


# Paths
ROOT_DIR = os.path.abspath("./")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "coco_weights/mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs_models")

# Dataset preparation
train_dataset = CocoLikeDataset()
train_dataset.load_data(
    '/content/drive/MyDrive/Mask-R-CNN-using-Tensorflow2-main/C&D-Management-3/train/_annotations.coco.json',
    '/content/drive/MyDrive/Mask-R-CNN-using-Tensorflow2-main/C&D-Management-3/train'
)
train_dataset.prepare()


# Display a sample image with masks
sample_id = train_dataset.image_ids[0]
sample_image = train_dataset.load_image(sample_id)
sample_mask, sample_class_ids = train_dataset.load_mask(sample_id)
sample_bboxes = extract_bboxes(sample_mask)

display_instances(
    sample_image, sample_bboxes, sample_mask, sample_class_ids,
    train_dataset.class_names
)

# Config & model setup
config = CDConfig()
config.display()

model = MaskRCNN(mode='training', model_dir=DEFAULT_LOGS_DIR, config=config)
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"
])

# Training
model.train(
    train_dataset, val_dataset,
    learning_rate=config.LEARNING_RATE,
    epochs=30,
    layers='heads'
)
