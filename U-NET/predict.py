from dataloader import load_images, load_masks, rgb2mask, get_image_names
from model import UnetFracture
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import time
from numpy import mean
# Load the YOLOv8 model
yolo_model = YOLO('best.pt')  # Replace 'best.pt' with the path to your YOLOv8 model

def apply_colormap(mask, num_classes=4):
    colormap = {
        0: [0, 0, 0],  # Class 0: Black (Background)
        1: [255, 255, 255],  # Class 1: Red
        2: [0, 255, 0],  # Class 2: Green
        3: [0, 0, 255],  # Class 3: Blue
    }
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in colormap.items():
        colored_mask[mask == class_idx] = color
    return colored_mask

# Load parameters
with open('params.json') as f:
    params = json.load(f)
model_params = params['shape_segmentation']['images_train_b8']

# Set mode and load data
mode = 'test'  # or 'train'
file_names = get_image_names(mode)
images = load_images(file_names, mode)
masks = load_masks(file_names, mode)

# Initialize the U-Net model
model = UnetFracture(model_params)
model.initialize()
inference_times = []
# Iterate through the dataset
for i, (image, mask_rgb) in enumerate(zip(images, masks)):
    mask_labels = rgb2mask(mask_rgb).astype(int)

    start_time = time.time()
    # YOLOv8 Prediction
    results = yolo_model.predict(image)
    mask_predict = model.predict_image(image)

    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000
    inference_times.append(inference_time_ms)
    boxes = results[0].boxes
    bounding_boxes = boxes.xyxy.cpu().numpy()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    items = ["Brick", "Concrete", "Tiles"]
    image_with_boxes = image.copy()

    # Draw bounding boxes
    for j, box in enumerate(bounding_boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image_with_boxes, (x_min, y_min), (x_max, y_max), colors[j % len(colors)], 1)
        cv2.putText(
            image_with_boxes,
            items[j % len(colors)],
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            colors[j % len(colors)],
            1,
            cv2.LINE_AA,
        )

    # U-Net Prediction


    # Apply colormap to U-Net prediction
    colored_mask_unet = apply_colormap(mask_predict)

    # Combine YOLO and U-Net results
    colored_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for j, box in enumerate(bounding_boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        color = colors[j % len(colors)]
        bbox_mask = mask_predict[y_min:y_max, x_min:x_max]
        for c in range(3):
            colored_mask[y_min:y_max, x_min:x_max, c] = np.where(
                bbox_mask == 1, color[c], colored_mask[y_min:y_max, x_min:x_max, c]
            )

    # Overlay the colored mask on the original image
    output_image = cv2.addWeighted(image, 0.8, colored_mask, 0.5, 0)
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Create a figure to display all three predictions in one image
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # YOLOv8 Prediction Image
    ax[0].imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')
    ax[0].set_title('YOLOv8 Predictions')

    # U-Net Prediction Image
    ax[1].imshow(colored_mask_unet)
    ax[1].axis('off')
    ax[1].set_title('U-Net Predicted Mask')

    # Combined YOLO and U-Net Image
    ax[2].imshow(cv2.cvtColor(output_image_rgb, cv2.COLOR_BGR2RGB))
    ax[2].axis('off')
    ax[2].set_title('Fused Output')

    # plt.tight_layout()
    # plt.show()
avg_inference_time_ms = mean(inference_times)
print(avg_inference_time_ms)
print('Prediction process completed.')
