from skimage import io
from skimage import measure
from skimage import transform
from skimage import color
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import cv2
from PIL import Image

# Compute the mean-squared error between two images
def MSE(srcpath, dstpath, scale = 256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    mse = measure.compare_mse(scr, dst)
    return mse

# Compute the normalized root mean-squared error (NRMSE) between two images
def NRMSE(srcpath, dstpath, mse_type = 'Euclidean', scale = 256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    nrmse = measure.compare_nrmse(scr, dst, norm_type = mse_type)
    return nrmse

# Compute the peak signal to noise ratio (PSNR) for an image
def PSNR(srcpath, dstpath, scale = 256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    psnr = measure.compare_psnr(scr, dst)
    return psnr

# Compute the mean structural similarity index between two images
def SSIM(srcpath, dstpath, RGBinput = True, scale = 256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    ssim = measure.compare_ssim(scr, dst, multichannel = RGBinput)
    return ssim

# Compute the mean L1 loss between two images
def L1Loss(srcpath, dstpath, scale = 256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    scr = cv2.imread(srcpath, cv2.IMREAD_GRAYSCALE) / 255.0
    dst = cv2.imread(dstpath, cv2.IMREAD_GRAYSCALE) / 255.0
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    l1loss = np.mean(np.abs(scr, dst))
    return l1loss

# Compute the mean L2 loss between two images
def L2Loss(srcpath, dstpath, scale = 256):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    scr = cv2.imread(srcpath, cv2.IMREAD_GRAYSCALE) / 255.0
    dst = cv2.imread(dstpath, cv2.IMREAD_GRAYSCALE) / 255.0
    scr = transform.resize(scr, (scale, scale))
    dst = transform.resize(dst, (scale, scale))
    l2loss = np.mean(np.square(scr, dst))
    return l2loss

# Compute the PrecisionScore between two images
def PrecisionScore(srcpath, dstpath, scale = 256):
    scr = cv2.imread(srcpath, cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread(dstpath, cv2.IMREAD_GRAYSCALE)
    scr = cv2.resize(scr, (scale, scale))
    dst = cv2.resize(dst, (scale, scale))
    scr = np.round(scr / 255.0).astype(np.int)
    dst = np.round(dst / 255.0).astype(np.int)
    scr = scr.flatten()
    dst = dst.flatten()
    pscore = precision_score(scr, dst)
    return pscore

# Compute the RecallScore between two images
def RecallScore(srcpath, dstpath, scale = 256):
    scr = cv2.imread(srcpath, cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread(dstpath, cv2.IMREAD_GRAYSCALE)
    scr = cv2.resize(scr, (scale, scale))
    dst = cv2.resize(dst, (scale, scale))
    scr = np.round(scr / 255.0).astype(np.int)
    dst = np.round(dst / 255.0).astype(np.int)
    scr = scr.flatten()
    dst = dst.flatten()
    rscore = recall_score(scr, dst)
    return rscore

# Compute the RecallScore between two images
def F1Score(srcpath, dstpath, scale = 256):
    scr = cv2.imread(srcpath, cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread(dstpath, cv2.IMREAD_GRAYSCALE)
    scr = cv2.resize(scr, (scale, scale))
    dst = cv2.resize(dst, (scale, scale))
    scr = np.round(scr / 255.0).astype(np.int)
    dst = np.round(dst / 255.0).astype(np.int)
    scr = scr.flatten()
    dst = dst.flatten()
    f1score = f1_score(scr, dst)
    return f1score

if __name__ == "__main__":
    srcpath = 'C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256_saliencymap\\ILSVRC2012_val_00000005.JPEG'
    dstpath = 'F:\\submitted papers\\my papers\\SCGAN v3\\major revision\\scgan saliency map, ILSVRC2012_val_256\\ILSVRC2012_val_00000005.png'
    pscore = PrecisionScore(srcpath, dstpath, scale = 256)
    print(pscore)
    rscore = RecallScore(srcpath, dstpath, scale = 256)
    print(rscore)
    f1score = F1Score(srcpath, dstpath, scale = 256)
    print(f1score)
    L1Loss = L1Loss(srcpath, dstpath, scale = 256)
    print(L1Loss)
    L2Loss = L2Loss(srcpath, dstpath, scale = 256)
    print(L2Loss)
