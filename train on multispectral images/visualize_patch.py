import cv2

folderlist = ['../uresnet no saliency map/val_results/', '../uresnet saliency map/val_results/', \
    '../pix2pix no saliency map/val_results/', '../pix2pix saliency map/val_results/', './val_results/']
imgnum = 1519

for i in range(6):
    if i < 5:
        imgname = folderlist[i] + str(imgnum) + '_pred.png'
        img = cv2.imread(imgname)
        img = img[90:140, 100:150]
        cv2.imwrite(str(i) + '.png', img)
    if i == 5:
        imgname = '../hrnet2/val_results/' + str(imgnum) + '_gt.png'
        img = cv2.imread(imgname)
        img = img[90:140, 100:150]
        cv2.imwrite('gt.png', img)
        imgname = '../hrnet2/val_results/' + str(imgnum) + '_in.png'
        img = cv2.imread(imgname)
        img = img[90:140, 100:150]
        cv2.imwrite('in.png', img)
'''
img = cv2.imread('../uresnet no saliency map/val_results/301_pred.png')
img = img[150:200, 95:145]
img = cv2.resize(img, (256, 256))
cv2.imshow('img', img)
cv2.waitKey(0)
'''
