from config import retinanet_config as config
import json
import os

csv_annotations = []
separator = ','


def convert_json_to_list():
    json_file = os.path.join('../' + config.JSON_PATH, 'all_labelled_20205013.json')
    with open(json_file) as json_file:
        images = json.load(json_file)
        for i, key in enumerate(images):
            image = images[key]
            image_path = os.path.join(config.IMAGES_PATH, key)
            labels = get_image_labels(image)
            if check_if_image_has_labels(labels):
                add_annotation(image_path, labels)
            else:
                add_annotation_without_label(image_path)


def get_image_labels(image):
    try:
        labels = image['labels']
    except KeyError:
        labels = []
    return labels


def check_if_image_has_labels(labels):
    return len(labels) > 0


def add_annotation_without_label(image_path):
    csv_annotations.append(image_path + ',,,\n')


def add_annotation(image_path, labels):
    for label in labels:
        geometry = label['geometry']
        min_cords = geometry[0]
        max_cors = geometry[1]
        x1 = min_cords[0]
        y1 = min_cords[1]
        x2 = max_cors[0]
        y2 = max_cors[1]
        csv_annotations.append(separator.join([image_path, str(x1), str(y1), str(x2), str(y2), 'sheep\n']))


def convert_list_to_csv():
    csv_file = os.path.join('../' + config.CSV_PATH, 'all_labeled.csv')
    with open(csv_file, 'w') as csv_file:
        csv_file.writelines(csv_annotations)


if __name__ == "__main__":
    convert_json_to_list()
    convert_list_to_csv()
