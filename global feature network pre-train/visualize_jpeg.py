import cv2
from PIL import Image
import numpy as np

'''
img = cv2.imread('example.JPEG')
jpeg_quality = 50

cv2.imwrite('temp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
compressed = cv2.imread('temp.jpg')

show = np.concatenate((img, compressed), axis = 1)
cv2.imshow('compare', show)
cv2.waitKey(0)
'''

img = Image.open('example.JPEG').convert('RGB')
jpeg_quality = 20
img.save('temp.jpg', quality = jpeg_quality)
img = np.array(img)

compressed = Image.open('temp.jpg').convert('RGB')
compressed = np.array(compressed)
show = np.concatenate((img, compressed), axis = 1)
show = Image.fromarray(show)
show.show()
