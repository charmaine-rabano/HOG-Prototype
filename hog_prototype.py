# Reference:
# https://data-flair.training/blogs/python-project-real-time-human-detection-counting/
#
# Only accepts image file as input

import cv2
import imutils

def detect(imageInput):
    image = cv2.imread(imageInput)

    image = imutils.resize(image, width = min(800, image.shape[1])) 
    
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(image, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    
    person = 1
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(image, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
    
    cv2.putText(image, f'People Count : {person-1}', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', image)
    print("People Count:",person-1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

imageInput = "people.jpg"
detect(imageInput)

