import math
import os
import random

import cv2
import pandas as pd
import inline as inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image



# https://www.youtube.com/watch?v=oXlwWbU8l2o
def preprocessing(methods,train_name,folder_name):
    images_path = './downloads/' + train_name + '/images'
    images_list= os.listdir(images_path)
    annotation_path = './downloads/'+train_name+'/labels'

    tr_name_flag = True
    for method in methods:
        print(method)
        if tr_name_flag:
            tr_name_flag = False
        else:
            images_path = 'preprocessing_folders/'+folder_name+'/images'
        for img in images_list:
            if img == 'dataset.yaml':
                continue
            print(img)
            image_name = img.split('.')[0]
            f = open(annotation_path + '/'+image_name+'.txt')
            yolo_annotations = f.readlines()
            f.close()

            image = cv2.imread(images_path+'/'+img)
            # Comvert BGR color from cv2 to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            annotations = convert_annotations(image, yolo_annotations)

            #image = draw_annotation(image, annotations)   #draw annotations

            # # Show image
            # im_pil = Image.fromarray(image)
            # im_pil.show()

            image = cv2.imread(images_path+'/'+ img)
            # Comvert BGR color from cv2 to RGB
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_pre, new_annotations,new_image_folder_path =  menu(method,image,annotations,folder_name)

            #image_pre = draw_annotation(image_pre, new_annotations) #draw annotations

            # # Show image
            # im_pil = Image.fromarray(image_pre)
            # im_pil.show()

            final_yolo_annotations = convert_to_yolo(image_pre, new_annotations)

            if not os.path.exists(new_image_folder_path):
                os.mkdir(new_image_folder_path)
            if not os.path.exists(new_image_folder_path+'/images'):
                os.mkdir(new_image_folder_path+'/images')
            if not os.path.exists(new_image_folder_path+'/labels'):
                os.mkdir(new_image_folder_path+'/labels')

            cv2.imwrite(new_image_folder_path+'/images/'+image_name+'.jpg', image_pre)
            if os.path.exists(new_image_folder_path+"/labels/"+image_name+".txt"):
                os.remove(new_image_folder_path+"/labels/"+image_name+".txt")
            for annotation in final_yolo_annotations:
                class_id, x, y, width, height = annotation
                line = class_id+' '+str(x)+' '+str(y)+' '+str(width)+' '+str(height)+'\n'
                f = open(new_image_folder_path+"/labels/"+image_name+".txt", "a")
                f.write(line)
                f.close()

        # # Show image
        # img2 = cv2.cvtColor(imag2, cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(img2)
        # im_pil.show()

    yaml_path = './preprocessing_folders/'+folder_name+'/'
    create_yaml(yaml_path)


import cv2
import numpy as np




def menu(fun_name,image, annotations,folder_name):


    if fun_name == 'get_Laplacian':
        image_pre, new_annotations = get_Laplacian(image, annotations)  # Give the image preprocessing technique ------------------------------------------------
        new_image_folder_path = 'preprocessing_folders/'+folder_name  # set name of folder
    elif fun_name == 'get_Sobel':
        image_pre, new_annotations = get_Sobel(image,annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name +'/'
    elif fun_name == 'get_grayscale':
        image_pre, new_annotations = get_grayscale(image,annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name +'/'
    elif fun_name == 'get_HSV':
        image_pre, new_annotations = get_HSV(image,annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_LAB':
        image_pre, new_annotations = get_LAB(image,annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_canny':
        image_pre, new_annotations = get_canny(image,annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_threshold':
        image_pre, new_annotations = get_threshold(image,annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_translation':
        dx = 200 #Give the right values
        dy = 100 #Give the right values
        image_pre, new_annotations = get_translation(image,annotations, dx, dy)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_scale':
        scale_factor = 0.3
        image_pre, new_annotations = get_scale(image,annotations,scale_factor)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_rotation_ccw':
        image_pre, new_annotations = get_rotation_ccw(image,annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_rotation_cw':
        image_pre, new_annotations = get_rotation_cw(image,annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_rotation':
        image_pre, new_annotations = get_rotation(image,annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_resize':
        scale_percent = 20
        image_pre, new_annotations = get_resize(image, annotations,scale_percent)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_crop':
        crop_size = 3000
        image_pre, new_annotations = get_crop(image, annotations, crop_size)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_blur':
        image_pre, new_annotations = get_blur(image, annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_gaussian_blur':
        image_pre, new_annotations = get_gaussian_blur(image, annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_median_blur':
        image_pre, new_annotations = get_median_blur(image, annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_bilateral_blur':
        image_pre, new_annotations = get_bilateral_blur(image, annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name
    elif fun_name == 'get_default_resize':
        image_pre, new_annotations = get_default_resize(image, annotations)
        new_image_folder_path = 'preprocessing_folders/' + folder_name

    return image_pre, new_annotations,new_image_folder_path

def create_yaml(yaml_path):
    data = {
        'path': yaml_path,
        'train': './images/',
        'val': './images/',
        'nc': 4,
        'names': ["Skoriasi", "Tetranychos", "Prasino Skouliki", "NoComment"]
    }

    with open(yaml_path+'dataset.yaml', 'w') as file:
        for key, value in data.items():
            if isinstance(value, list):
                value = [str(item) for item in value]
                value = '[' + ', '.join(value) + ']'
            file.write(f'{key}: {value}\n')


#--------Annotation
def convert_annotations(image,yolo_annotations):
    # convert annotations to pixel coordinates
    img_h, img_w = image.shape[:2]
    pixel_annotations = []
    for annotation in yolo_annotations:
        try:
            class_id, x, y, width, height = annotation.split()
        except:
            class_id, x, y, width, height = annotation

        x = int(float(x) * img_w)
        y = int(float(y) * img_h)
        width = int(float(width) * img_w)
        height = int(float(height) * img_h)
        pixel_annotations.append((class_id, x, y, width, height))

    annotations = []
    for annotation in pixel_annotations:
        class_id, x, y, width, height = annotation
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        annotations.append((class_id,x1,y1,x2,y2))

    return annotations

def convert_to_yolo(image, annotations):

    # convert annotations to YOLOv5 format
    img_h, img_w = image.shape[:2]
    yolo_annotations = []
    for annotation in annotations:

        class_id, x1, y1, x2, y2 = annotation
        x = (x1 + x2) / 2 / img_w
        y = (y1 + y2) / 2 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        yolo_annotations.append((class_id, x, y, width, height))
    return yolo_annotations

def draw_annotation(image,annotations):

    # draw annotations on image
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return image


#--------Edge detection
def get_Laplacian(image,annotations):
    gray = get_grayscale(image,annotations)
    image = cv2.Laplacian(gray[0], cv2.CV_64F)
    image = np.uint8(np.absolute(image))

    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id,x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))

    return image,new_annotations

def get_Sobel(image,annotations):
    gray = get_grayscale(image,annotations)
    y = cv2.Sobel(gray[0], cv2.CV_64F,0,1)
    x = cv2.Sobel(gray[0], cv2.CV_64F, 1, 0)

    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))

    return cv2.bitwise_or(x,y) , new_annotations
#--------Color spaces
def get_grayscale(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),new_annotations

def get_HSV(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))

    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV),new_annotations

def get_LAB(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))

    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV),new_annotations

#--------Contours
def get_canny(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))

    #image =  cv2.blur(image, (5, 5))
    return cv2.Canny(image, 125,175),new_annotations

def get_threshold(image,annotations):
    gray = get_grayscale(image,annotations)

    ret,thresh = cv2.threshold(gray[0],125,255,cv2.THRESH_BINARY)

    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))

    return thresh,new_annotations

#--------Transformation

def get_translation(image, annotations, dx, dy):
    #dx =200 , dy =100
    # Translate the image using the specified amounts of dx and dy
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_image = cv2.warpAffine(image, M, (cols, rows))

    # Update the annotations based on the image translation
    translated_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        x1 += dx
        x2 += dx
        y1 += dy
        y2 += dy
        translated_annotations.append((class_id, x1, y1, x2, y2))

    return translated_image, translated_annotations

def get_scale(image, annotations, scale_factor):
    # Get image dimensions
    height, width = image.shape[:2]

    # Scale the image
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    # Scale the annotations
    scaled_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        scaled_x1 = int(x1 * scale_factor)
        scaled_y1 = int(y1 * scale_factor)
        scaled_x2 = int(x2 * scale_factor)
        scaled_y2 = int(y2 * scale_factor)
        scaled_annotations.append((class_id, scaled_x1, scaled_y1, scaled_x2, scaled_y2))

    return scaled_image, scaled_annotations

def get_rotation_ccw(image, annotations):
    # get image height and width
    img_h, img_w = image.shape[:2]

    # rotate image
    rotated_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # rotate annotation coordinates
    rotated_ann = []
    for ann in annotations:
        class_id, x1, y1, x2, y2 = ann
        new_x1 = y1
        new_y1 = img_w - x2
        new_x2 = y2
        new_y2 = img_w - x1
        rotated_ann.append((class_id, new_x1, new_y1, new_x2, new_y2))

    return rotated_img, rotated_ann

def get_rotation_cw(image, annotations):
    # get image dimensions
    img_h, img_w = image.shape[:2]

    # rotate image
    rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # rotate annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_x1 = img_h - y2
        new_y1 = x1
        new_x2 = img_h - y1
        new_y2 = x2
        new_annotations.append((class_id, new_x1, new_y1, new_x2, new_y2))

    return rotated_img, new_annotations

def get_rotation(image, annotations):
    # rotate the image
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)

    # rotate the annotations
    rotated_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_x1 = rotated_image.shape[1] - x1
        new_y1 = rotated_image.shape[0] - y1
        new_x2 = rotated_image.shape[1] - x2
        new_y2 = rotated_image.shape[0] - y2
        rotated_annotations.append((class_id, new_x2, new_y2, new_x1, new_y1))

    return rotated_image, rotated_annotations

def get_resize(image, annotations, scale_percent):
    # Get image dimensions
    img_h, img_w = image.shape[:2]

    # Scale the image
    width = int(img_w * scale_percent / 100)
    height = int(img_h * scale_percent / 100)
    image = cv2.resize(image, (width, height))

    # Scale the annotations accordingly
    scaled_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        x1 = int(x1 * (width / img_w))
        y1 = int(y1 * (height / img_h))
        x2 = int(x2 * (width / img_w))
        y2 = int(y2 * (height / img_h))
        scaled_annotations.append((class_id, x1, y1, x2, y2))

    return image, scaled_annotations


def get_default_resize(image, annotations, width=640, height=640):
    # Get original image dimensions
    img_h, img_w = image.shape[:2]

    # Resize the image
    image = cv2.resize(image, (width, height))

    # Scale the annotations accordingly
    scaled_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        x1 = int(x1 * (width / img_w))
        y1 = int(y1 * (height / img_h))
        x2 = int(x2 * (width / img_w))
        y2 = int(y2 * (height / img_h))
        scaled_annotations.append((class_id, x1, y1, x2, y2))

    return image, scaled_annotations

def get_crop(img, annotations, crop_size):

    #:param crop_size: size of the cropped image for example crop_size=3000

    # Get image dimensions
    height, width = img.shape[:2]

    # Randomly select a crop location
    x = random.randint(0, width - crop_size)
    y = random.randint(0, height - crop_size)

    # Crop the image
    crop_img = img[y:y + crop_size, x:x + crop_size]

    # Adjust the annotations based on the crop location
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        # Check if annotation is completely inside the crop region
        if x <= x1 and x + crop_size >= x2 and y <= y1 and y + crop_size >= y2:
            new_x1, new_y1, new_x2, new_y2 = x1 - x, y1 - y, x2 - x, y2 - y
            new_annotations.append((class_id, new_x1, new_y1, new_x2, new_y2))
        # Check if annotation is partially inside the crop region
        elif x <= x1 < x + crop_size or x <= x2 < x + crop_size:
            new_x1 = max(x1 - x, 0)
            new_x2 = min(x2 - x, crop_size)
            if y <= y1 < y + crop_size:
                new_y1 = max(y1 - y, 0)
                new_y2 = min(y2 - y, crop_size)
                new_annotations.append((class_id, new_x1, new_y1, new_x2, new_y2))
            elif y <= y2 < y + crop_size:
                new_y1 = max(y2 - y, 0)
                new_y2 = min(y2 - y + (y2 - y1), crop_size)
                new_annotations.append((class_id, new_x1, new_y1, new_x2, new_y2))
        # Check if annotation is completely outside the crop region
        elif x1 >= x + crop_size or x2 <= x or y1 >= y + crop_size or y2 <= y:
            continue

    return crop_img, new_annotations

#--------Blur
def get_blur(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))
    #Avarage blur
    return cv2.blur(image, (10, 10)),new_annotations

def get_gaussian_blur(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))

    return cv2.GaussianBlur(image,(7,7),0),new_annotations

def get_median_blur(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))

    #best to remove noise
    return cv2.medianBlur(image,5),new_annotations

def get_bilateral_blur(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        class_id, x1, y1, x2, y2 = annotation
        new_annotations.append((class_id, x1, y1, x2, y2))

    #blur with keeping edges
    return cv2.bilateralFilter(image,5,15,15),new_annotations

