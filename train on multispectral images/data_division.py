import os
import cv2
import numpy as np

def get_folderpath(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            wholepath = os.path.join(root, filespath)
            foldersubpath = wholepath.split('\\')[:-1]
            folderpath = ''
            for i in range(len(foldersubpath)):
                folderpath = os.path.join(folderpath, foldersubpath[i])
            ret.append(folderpath)
    return ret

def get_folderpath_once(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            wholepath = os.path.join(root, filespath)
            foldersubpath = wholepath.split('\\')[:-1]
            folderpath = ''
            for i in range(len(foldersubpath)):
                folderpath = os.path.join(folderpath, foldersubpath[i])
            if folderpath not in ret:
                ret.append(folderpath)
    return ret

def get_half_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            wholepath = os.path.join(root, filespath)
            if wholepath.split('\\')[-2] == 'visible':
                ret.append(wholepath)
    return ret

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
readpath = 'E:\\dataset, task related\\Multi Spectral\\KAIST dataset'
trainpath = 'E:\\dataset, task related\\Multi Spectral\\KAIST dataset processed\\train'
valpath = 'E:\\dataset, task related\\Multi Spectral\\KAIST dataset processed\\val'

filelist = get_half_files(readpath)

for i, item in enumerate(filelist):
    print(i, item)

    # path processing
    jpgname = item.split('\\')[-1]
    subfolder_name = item.split('\\')[-3]
    set_name = item.split('\\')[-4]
    train_lwir_savepath = os.path.join(trainpath, set_name, subfolder_name, 'lwir', jpgname)
    train_visible_savepath = os.path.join(trainpath, set_name, subfolder_name, 'visible', jpgname)
    val_lwir_savepath = os.path.join(valpath, set_name, subfolder_name, 'lwir', jpgname)
    val_visible_savepath = os.path.join(valpath, set_name, subfolder_name, 'visible', jpgname)
    lwir_img_path = os.path.join(readpath, set_name, subfolder_name, 'lwir', jpgname)
    visible_img_path = os.path.join(readpath, set_name, subfolder_name, 'visible', jpgname)

    # read images
    lwir_img = cv2.imread(lwir_img_path)
    visible_img = cv2.imread(visible_img_path)

    # save
    if i % 20 == 0:
        check_path(os.path.join(trainpath, set_name, subfolder_name, 'lwir'))
        check_path(os.path.join(trainpath, set_name, subfolder_name, 'visible'))
        cv2.imwrite(val_lwir_savepath, lwir_img)
        cv2.imwrite(val_visible_savepath, visible_img)
    else:
        check_path(os.path.join(valpath, set_name, subfolder_name, 'lwir'))
        check_path(os.path.join(valpath, set_name, subfolder_name, 'visible'))
        cv2.imwrite(train_lwir_savepath, lwir_img)
        cv2.imwrite(train_visible_savepath, visible_img)
