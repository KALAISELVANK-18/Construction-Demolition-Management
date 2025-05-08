
import datetime
import matplotlib.patches as patches
import skimage.io
import skimage.color

from numpy import expand_dims, mean

from mrcnn.model import MaskRCNN, load_image_gt, mold_image
from mrcnn.utils import compute_ap
from mrcnn import visualize


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




# Configuration for inference
class PredictionConfig(Config):
    NAME = "c&d"
    NUM_CLASSES = 1 + 3
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 15

# mAP evaluation with visualization
def evaluate_model(dataset, model, cfg):
    APs = []
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id)
        results = model.detect([image], verbose=0)[0]

        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset.class_names, title="Ground Truth")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        visualize.display_instances(image, results['rois'], results['masks'], results['class_ids'],
                                    dataset.class_names, results['scores'], title="Prediction")
        plt.axis('off')

        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 results["rois"], results["class_ids"], results["scores"], results["masks"])
        print(AP)
        if AP > 0.7:
            APs.append(AP)
    return mean(APs)

# Display image with ground truth annotations
def display_image_and_annotations(image, gt_class_id, gt_bbox, gt_mask):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    for i in range(gt_bbox.shape[0]):
        y1, x1, y2, x2 = gt_bbox[i]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        class_id = gt_class_id[i]
        ax.text(x1, y1 - 2, f'Class: {class_id}', color='white', fontsize=12)
    for i in range(gt_mask.shape[-1]):
        mask = gt_mask[:, :, i]
        masked_image = np.ma.masked_where(mask == 0, mask)
        ax.imshow(masked_image, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

# Color splash effect
def color_splash(img, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(img)) * 255
    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        return np.where(mask, img, gray).astype(np.uint8)
    return gray.astype(np.uint8)

# Detect and apply color splash on image/video
def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path
    if image_path:
        img = skimage.io.imread(image_path)
        r = model.detect([img], verbose=1)[0]
        splash = color_splash(img, r['masks'])
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
        success, image = vcapture.read()
        while success:
            image = image[..., ::-1]
            r = model.detect([image], verbose=0)[0]
            splash = color_splash(image, r['masks'])
            splash = splash[..., ::-1]
            vwriter.write(splash)
            success, image = vcapture.read()
        vwriter.release()


val_dataset = CocoLikeDataset()
val_dataset.load_data(
    'C&D-Management-3/valid/_annotations.coco.json',
    'C&D-Management-3/valid')


val_dataset.prepare()


inference_config = PredictionConfig()
inference_model = MaskRCNN(mode='inference', config=inference_config, model_dir='./logs')
inference_model.load_weights('mask_rcnn_c&d waste management_0040.h5', by_name=True)

evaluate_model(val_dataset, inference_model, inference_config)