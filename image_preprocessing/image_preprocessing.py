import os
import cv2
import inline as inline
import matplotlib
import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt



def preprocessing():
    images_path= os.listdir('./images')
    for img in images_path:
        image = cv2.imread('./images/'+img, 1)

        imag2 = get_grayscale(image)
        #Show image
        # cv2.imshow(image, img)
        # k = cv2.waitKey(0)
        # if k == 27 or k == ord('q'):
        #     cv2.destroyAllWindows()


        cv2.imwrite('new.jpg', imag2)





def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)













preprocessing()