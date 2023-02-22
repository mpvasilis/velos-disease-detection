classes = myDict = {"Σκωρίαση": '0', "Τετράνυχος": '1', 'Πράσινο σκουλήκι' : '2' }

def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [str(((2*x1 + w)/(2*image_w))) , str(((2*y1 + h)/(2*image_h))), str(w/image_w), str(h/image_h)]

