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
        anno_data = f.readlines()
        f.close()



        image = cv2.imread('../downloads/train/images/'+img)


        get_Annotation(image, anno_data)

        exit()

        #Comvert BGR color from cv2 to RGB
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


        imag2 = get_grayscale(image)

        # Show image
        img2 = cv2.cvtColor(imag2, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img2)
        im_pil.show()

        # cv2.imshow('Image',imag2)
        # cv2.waitKey(0)

        exit()

        cv2.imwrite('new.jpg', imag2)


#--------Annotation
def get_Annotation(image,anno_data):
    imageRectangle = image.copy()

    dh,dw,_ = image.shape

    for a in anno_data:
        _, x, y, w, h = map(float, a.split(' '))
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1
        cv2.rectangle(imageRectangle, (l, t), (r, b), (0, 0, 255), 3)


    img2 = cv2.cvtColor(imageRectangle, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img2)
    im_pil.show()

    return imageRectangle



#--------Edge detection
def get_Laplacian(image):
    gray = get_grayscale(image)
    image = cv2.Laplacian(gray, cv2.CV_64F)
    return np.uint8(np.absolute(image))

def get_Sobel(image):
    gray = get_grayscale(image)
    y = cv2.Sobel(gray, cv2.CV_64F,0,1)
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    return cv2.bitwise_or(x,y)
#--------Color spaces
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def get_HSV(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def get_LAB(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

#--------Contours
def get_canny(image):
    #image =  cv2.blur(image, (5, 5))
    return cv2.Canny(image, 125,175)
def get_threshold(image):
    image2 = get_grayscale(image)

    ret,thresh = cv2.threshold(image2,125,255,cv2.THRESH_BINARY)
    return thresh

#--------Transformation
def get_scale(image,height,width):
    return cv2.resize(image, (height,width), interpolation=cv2.INTER_CUBIC)

def get_translation(image):
    rows = image.shape[1]
    cols = image.shape[0]
    M = np.float32([[1, 0, 200], [0, 1, 100]])
    return cv2.warpAffine(image, M, (rows, cols))

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
