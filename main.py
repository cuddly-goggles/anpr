import cv2
import os.path


def predict(img, model):
    classes, confidences, boxes  = model.detect(img, 0.4, 0.2)
    if len(classes) == 0:
        return 2
    if len(classes) > 1:
        return int(np.where(confidences == max(confidences))[0])
    if len(classes) == 1:
        return int(classes)




def main():
    net = cv2.dnn.readNet('weights/yolov4-lp2_best.weights', 'yolov4-lp2.cfg')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(608, 608), scale=1 / 255)

    dataset_file = open('dataset.txt', 'r')
    images = dataset_file.read().split('\n')
    dataset_file.close()
    predictions = []
    print(images)
    for image in images:
        if not os.path.isfile(image):
            images.remove(image)
            print(image, ' is not file. REMOVED FROM ARRAY')
            continue
        print('Predicting ', image)
        try:
            image = cv2.imread(image)
        except cv2.error:
            images.remove(image)
            print(image, ' OPENCV can not read this file. REMOVED FROM ARRAY')
            continue
        predictions.append(predict(image, model))

    predictions_file = open('predictions.txt', 'w')
    for i in range(len(predictions)):
        predictions_file.write(f'{images[i]}, {predictions[i]}\n')
    predictions_file.close()


if __name__ == '__main__':
    main()