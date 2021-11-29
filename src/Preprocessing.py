import cv2 as cv
import os

ORIGINAL_PATH = "C:/Users/Public/Pictures/Original_picture/Children"
DISCERN_UPLOAD_PATH = "C:/Users/Public/Pictures/Preprocess_dataset/children"
CASCADE_CLASSIFIER_PATH = "setting/haarcascade_frontalface_default.xml"
face_detect = cv.CascadeClassifier(CASCADE_CLASSIFIER_PATH)


def face_detect_fun(img, img_path, index):
    # convert the colour RGB image to a grayscale luminance image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face = face_detect.detectMultiScale(gray)
    for x, y, w, h in face:
        roi = gray[y:y + h, x:x + w]
        re_roi = cv.resize(roi, (48, 48))
        # rename each picture
        if index < 10:
            str_index = '000' + str(index)
        elif index < 100:
            str_index = '00' + str(index)
        elif index < 1000:
            str_index = '0' + str(index)
        else:
            str_index = str(index)
        save_path = os.path.join(DISCERN_UPLOAD_PATH, "Child_" + str_index + '.jpg')
        cv.imwrite(save_path, re_roi)


def read_img_batch(path, endpoint=None):
    container = []
    index = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            input_img = cv.imread(path)
            # convert the input image to grayscale and scale down it to 1/4 of the original size.
            face_detect_fun(input_img, path, index)
            index += 1
    return container


read_img_batch(ORIGINAL_PATH)
