import cv2
import glob
import random
import numpy as np
import os

def save_detected_image(img, filename):
    if 'b' in filename:
        cv2.imwrite(f'detected/1/{filename}', img)
    if 'c' in filename:
        cv2.imwrite(f'detected/2/{filename}', img)
    if 'b' not in filename and 'c' not in filename:
        cv2.imwrite(f'detected/0/{filename}', img)


def main():
    net = cv2.dnn.readNet('weights/yolov4-lp2_best.weights', 'yolov4-lp2.cfg')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(608, 608), scale=1/255)

    zero_images = glob.glob('alldataset/0/*.jpg')
    one_images = glob.glob('alldataset/1/*.jpg')
    two_images = glob.glob('alldataset/2/*.jpg')
    all_images = zero_images + one_images + two_images
    random.shuffle(all_images)

    for image in all_images:
        img = cv2.imread(image)
        classes, confidences, boxes  = model.detect(img, 0.2, 0.2)
        if len(classes) == 0:
            save_detected_image(img, os.path.basename(image))
            continue
        for class_id, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            label = '%.2f' % confidence
            label = '%s: %s' % (names[class_id], label)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            left, top, width, height = box
            top = max(top, labelSize[1])
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.rectangle(img, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        save_detected_image(img, os.path.basename(image))


if __name__ == '__main__':
    main()