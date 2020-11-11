import cv2
import numpy as np

lwir_path = 'E:\\dataset, task related\\Multi Spectral\\KAIST dataset processed\\train\\set00\\V000\\lwir\\I00001.jpg'
visible_path = 'E:\\dataset, task related\\Multi Spectral\\KAIST dataset processed\\train\\set00\\V000\\visible\\I00001.jpg'
saliency_path = 'E:\\dataset, task related\\Multi Spectral\\KAIST dataset processed\\train\\set00\\V000\\saliency map\\I00001.jpg'

lwir_img = cv2.imread(lwir_path)
visible_img = cv2.imread(visible_path)
saliency_img = cv2.imread(saliency_path)
print(lwir_img.shape)           # (512, 640, 3)
print(visible_img.shape)        # (512, 640, 3)
print(saliency_img.shape)       # (512, 640, 3)

concat_img = np.concatenate((lwir_img, visible_img, saliency_img), axis = 1)
cv2.imshow('concat_img', concat_img)
cv2.waitKey(0)
