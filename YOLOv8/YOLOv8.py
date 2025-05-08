import os
import shutil
import subprocess
import json
from ultralytics import YOLO, checks

def check_gpu():
    subprocess.run(["nvidia-smi"])

def get_home():
    home = os.getcwd()
    print(f"Working Directory: {home}")
    return home

def verify_ultralytics():
    checks()

def predict_default_image():
    subprocess.run([
        "yolo", "task=detect", "mode=predict",
        "model=yolov8n.pt", "conf=0.25",
        "source=https://media.roboflow.com/notebooks/examples/dog.jpeg",
        "save=True"
    ])

def download_dataset():
    from roboflow import Roboflow
    rf = Roboflow(api_key="dqxWTQU59hN3RPLyK6KK")
    project = rf.workspace("cd-waste-management").project("c-d-management")
    version = project.version(3)
    return version.download("coco")

def train_model(data_yaml_path):
    subprocess.run([
        "yolo", "task=detect", "mode=train",
        "model=yolov8s.pt", f"data={data_yaml_path}",
        "epochs=100", "imgsz=640"
    ])

def convert_coco_to_yolo(coco_annotation_file, images_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(coco_annotation_file) as f:
        coco = json.load(f)

    images_info = {img['id']: img for img in coco['images']}

    def coco_to_yolo_bbox(bbox, img_width, img_height):
        x_min, y_min, width, height = bbox
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        width /= img_width
        height /= img_height
        return [x_center, y_center, width, height]

    for annotation in coco['annotations']:
        img_id = annotation['image_id']
        img_info = images_info[img_id]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        bbox_yolo = coco_to_yolo_bbox(annotation['bbox'], img_width, img_height)
        class_id = annotation['category_id'] - 1
        img_basename = os.path.splitext(img_filename)[0]
        yolo_annotation_file = os.path.join(output_dir, f"{img_basename}.txt")

        with open(yolo_annotation_file, 'a') as f:
            bbox_str = " ".join(map(str, bbox_yolo))
            f.write(f"{class_id} {bbox_str}\n")

    print("Conversion from COCO to YOLO format is complete!")

def remove_old_labels(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def backup_folder(source_folder, destination_folder):
    if os.path.commonpath([source_folder]) == os.path.commonpath([source_folder, destination_folder]):
        raise ValueError("The destination folder cannot be inside the source folder.")
    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)
    os.chdir(destination_folder)
    print("Current working directory:", os.getcwd())

def predict_with_custom_model(model_path, source_folder):
    subprocess.run([
        "yolo", "task=detect", "mode=predict",
        f"model={model_path}", "conf=0.25",
        f"source={source_folder}", "save=True"
    ])

def validate_model(model_path, data_yaml_path):
    model = YOLO(model_path)
    results = model.val(data=data_yaml_path)

    print(f"\nmAP@0.5: {results.maps[0]:.4f}")
    print(f"mAP@0.5:0.95: {results.maps.mean():.4f}\n")

    for i, class_name in enumerate(results.names):
        precision = results.class_result(i)[1]
        recall = results.class_result(i)[2]
        f1 = results.class_result(i)[3]
        print(f"{class_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1:.4f}")

    mean_precision = results.box.p.mean()
    mean_recall = results.box.r.mean()
    mean_f1 = results.box.f1.mean()

    print(f"\nMean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1 score: {mean_f1:.4f}")

if __name__ == "__main__":
    check_gpu()
    HOME = get_home()
    verify_ultralytics()
    predict_default_image()

    # Dataset & Training
    dataset = download_dataset()
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    train_model(data_yaml_path)

    # COCO to YOLO conversion
    coco_annotation_file = "/path/to/annotations/_annotations.coco.json"
    images_dir = "/path/to/images"
    output_dir = "/path/to/output_yolo_labels"
    convert_coco_to_yolo(coco_annotation_file, images_dir, output_dir)

    # Remove old labels (optional cleanup)
    remove_old_labels("/path/to/old/labels")

    # Predict on test images
    best_model_path = "/path/to/best.pt"
    test_images_path = "/path/to/test/images"
    predict_with_custom_model(best_model_path, test_images_path)

    # Backup training output (optional)
    backup_folder("/path/to/source", "/path/to/destination")

    # Validate model
    validate_model(best_model_path, data_yaml_path)
