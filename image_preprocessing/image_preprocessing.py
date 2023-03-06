import os
import cv2
import pandas as pd
import inline as inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image



# https://www.youtube.com/watch?v=oXlwWbU8l2o
def preprocessing():
    images_path= os.listdir("../downloads/train/images")
    annotation_path = '../downloads/train/labels'
    for img in images_path:
        image_name = img.split('.')[0]
        f = open(annotation_path + '/'+image_name+'.txt')
        annotations = f.readlines()
        f.close()

        image = cv2.imread('../downloads/train/images/'+img)
        # Comvert BGR color from cv2 to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = get_Annotation(image, annotations)

        # Show image
        im_pil = Image.fromarray(image)
        im_pil.show()


        image = cv2.imread('../downloads/train/images/' + img)
        # Comvert BGR color from cv2 to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pre,new_annotations = get_canny(image,annotations)    #Give the image preprocessing technique ------------------------------------------------


        image_pre = get_Annotation(image_pre, new_annotations)
        # Show image
        im_pil = Image.fromarray(image_pre)
        im_pil.show()



        new_image_folder_path = 'new_data/get_Laplacian'   #set name of folder
        if not os.path.exists(new_image_folder_path):
            os.mkdir(new_image_folder_path)
        if not os.path.exists(new_image_folder_path+'/images'):
            os.mkdir(new_image_folder_path+'/images')
        if not os.path.exists(new_image_folder_path+'/labels'):
            os.mkdir(new_image_folder_path+'/labels')

        cv2.imwrite(new_image_folder_path+'/images/'+image_name+'.jpg', image_pre)

        exit()

        # # Show image
        # img2 = cv2.cvtColor(imag2, cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(img2)
        # im_pil.show()


import cv2
import numpy as np



#--------Annotation
def get_Annotation(image,annotations):
    # convert annotations to pixel coordinates
    img_h, img_w = image.shape[:2]
    pixel_annotations = []
    for annotation in annotations:
        try:
            class_id, x, y, width, height = annotation.split()
        except:
            class_id, x, y, width, height = annotation[:]

        x = int(float(x) * img_w)
        y = int(float(y) * img_h)
        width = int(float(width) * img_w)
        height = int(float(height) * img_h)
        pixel_annotations.append((class_id, x, y, width, height))

    # draw annotations on image
    for annotation in pixel_annotations:
        class_id, x, y, width, height = annotation
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return image


#--------Edge detection
def get_Laplacian(image,annotations):
    gray = get_grayscale(image)
    image = cv2.Laplacian(gray, cv2.CV_64F)
    image = np.uint8(np.absolute(image))

    # resize annotations
    new_annotations = []
    for annotation in annotations:
        annotation = annotation.split(' ')
        annotation[0] = int(annotation[0])
        annotation[1] = float(annotation[1])
        annotation[2] = float(annotation[2])
        annotation[3] = float(annotation[3])
        annotation[4] = float(annotation[4].strip())

        x1, y1, width, height = annotation[1:]
        new_annotations.append([annotation[0], x1, y1, width, height])
    return image,new_annotations

def get_Sobel(image,annotations):
    gray = get_grayscale(image)
    y = cv2.Sobel(gray, cv2.CV_64F,0,1)
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    # resize annotations
    new_annotations = []
    for annotation in annotations:
        annotation = annotation.split(' ')
        annotation[0] = int(annotation[0])
        annotation[1] = float(annotation[1])
        annotation[2] = float(annotation[2])
        annotation[3] = float(annotation[3])
        annotation[4] = float(annotation[4].strip())

        x1, y1, width, height = annotation[1:]
        new_annotations.append([annotation[0], x1, y1, width, height])

    return cv2.bitwise_or(x,y) , new_annotations
#--------Color spaces
def get_grayscale(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        annotation = annotation.split(' ')
        annotation[0] = int(annotation[0])
        annotation[1] = float(annotation[1])
        annotation[2] = float(annotation[2])
        annotation[3] = float(annotation[3])
        annotation[4] = float(annotation[4].strip())

        x1, y1, width, height = annotation[1:]
        new_annotations.append([annotation[0], x1, y1, width, height])

    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),new_annotations

def get_HSV(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        annotation = annotation.split(' ')
        annotation[0] = int(annotation[0])
        annotation[1] = float(annotation[1])
        annotation[2] = float(annotation[2])
        annotation[3] = float(annotation[3])
        annotation[4] = float(annotation[4].strip())

        x1, y1, width, height = annotation[1:]
        new_annotations.append([annotation[0], x1, y1, width, height])

    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV),new_annotations

def get_LAB(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        annotation = annotation.split(' ')
        annotation[0] = int(annotation[0])
        annotation[1] = float(annotation[1])
        annotation[2] = float(annotation[2])
        annotation[3] = float(annotation[3])
        annotation[4] = float(annotation[4].strip())

        x1, y1, width, height = annotation[1:]
        new_annotations.append([annotation[0], x1, y1, width, height])

    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV),new_annotations

#--------Contours
def get_canny(image,annotations):
    # resize annotations
    new_annotations = []
    for annotation in annotations:
        annotation = annotation.split(' ')
        annotation[0] = int(annotation[0])
        annotation[1] = float(annotation[1])
        annotation[2] = float(annotation[2])
        annotation[3] = float(annotation[3])
        annotation[4] = float(annotation[4].strip())

        x1, y1, width, height = annotation[1:]
        new_annotations.append([annotation[0], x1, y1, width, height])

    #image =  cv2.blur(image, (5, 5))
    return cv2.Canny(image, 125,175),new_annotations


def get_threshold(image,annotations):
    image2 = get_grayscale(image)

    ret,thresh = cv2.threshold(image2,125,255,cv2.THRESH_BINARY)

    # resize annotations
    new_annotations = []
    for annotation in annotations:
        annotation = annotation.split(' ')
        annotation[0] = int(annotation[0])
        annotation[1] = float(annotation[1])
        annotation[2] = float(annotation[2])
        annotation[3] = float(annotation[3])
        annotation[4] = float(annotation[4].strip())

        x1, y1, width, height = annotation[1:]
        new_annotations.append([annotation[0], x1, y1, width, height])

    return thresh,new_annotations

#--------Transformation

def get_translation(image, annotations, tx, ty):
    #tx =200 , ty =100
    # define translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    # translate image
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

    new_annotations = []
    for annotation in annotations:
        class_id, x, y, width, height = annotation.split()
        x = float(x)
        y = float(y)
        width = float(width)
        height = float(height)

        # scale coordinates to pixel coordinates
        img_h, img_w = image.shape[:2]
        x = int(x * img_w)
        y = int(y * img_h)
        width = int(width * img_w)
        height = int(height * img_h)

        # translate coordinates
        x += int(tx)
        y += int(ty)

        # scale coordinates back to original coordinates
        x = x / img_w
        y = y / img_h
        width = width / img_w
        height = height / img_h

        # append updated annotation
        new_annotations.append("{} {} {} {} {}".format(class_id, x, y, width, height))

    return translated_image, new_annotations

def get_scale(image,height,width):
    return cv2.resize(image, (height,width), interpolation=cv2.INTER_CUBIC)


def get_rotation_ccw(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

def get_rotation_cw(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def get_rotation(image):
    return cv2.rotate(image, cv2.ROTATE_180)

def get_resize_dimentions(image,height,width):
    return cv2.resize(image, (height,width))

def get_resize(image):
    return cv2.resize(image, None, fx=0.10, fy=0.10)

def get_rotate(image,angle,rotPoint=None):
    (height,width) = image.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2,height//2)
    rotMat=cv2.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions = (width,height)
    return cv2.warpAffine(image, rotMat, dimensions)

def get_crop(image):
    rows = image.shape[1]
    cols = image.shape[0]
    hgt, wdt = image.shape[:2]
    start_row, start_col = int(hgt * .25), int(wdt * .25)
    end_row, end_col = int(cols * .75), int(rows * .75)
    return image[start_row:end_row, start_col:end_col]

#--------Blur
def get_blur(image):
    #Avarage blur
    return cv2.blur(image, (10, 10))

def get_gaussian_blur(image):
    return cv2.GaussianBlur(image,(7,7),0)


def get_median_blur(image):
    #best to remove noise
    return cv2.medianBlur(image,5)

def get_bilateral_blur(image):
    #blur with keeping edges
    return cv2.bilateralFilter(image,5,15,15)

#--------Histogram
def show_histogram(image):
    #Plot histogram
    img_hist = image / 255
    pd.Series(img_hist.flatten()).plot(kind='hist',bins=50,)
    plt.show()


#--------RGB chanels

def show_RGB_chanels(image):
    # Show RGB
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image[:, :, 0], cmap='Reds')
    axs[1].imshow(image[:, :, 1], cmap='Greens')
    axs[2].imshow(image[:, :, 2], cmap='Blues')
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    plt.show()

preprocessing()
