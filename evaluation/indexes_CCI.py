from skimage import io
from skimage import color
import numpy as np
import math

# single image transformation from rgb to hsl
def rgb2hsl(r, g, b):                                       # for rgb2hsl color transformation
    r = r / 255
    g = g / 255
    b = b / 255
    Cmax = max(r, g, b)
    Cmin = min(r, g, b)
    pf = Cmax + Cmin                                        # no more than 2   [0, 2]
    df = Cmax - Cmin                                        # bigger than or equal to 0   [0, 1]
    # compute h   [0, 360]
    if Cmax == Cmin:
        h = 0
    elif Cmax == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif Cmax == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif Cmax == b:
        h = (60 * ((r-g)/df) + 240) % 360
    # compute l   [0, 1]
    l = pf / 2
    # compute s   [0, 1]
    if df == 0:
        s = 0
    else:
        s = df/(1 - abs(2 * l - 1))
    return h, s, l

# compute the average value of a list
def averagenum(num):
    nsum = 0
    if len(num) > 0:
        for i in range(len(num)):
            nsum += num[i]
        nsum = nsum / len(num)
    else:
        nsum
    return nsum

# Compute the color naturalness index (CNI) for one image
def SingleImageCNI(path):
    # Step 1: read the image and transform it to HSL color space
    img = io.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    # Step 2: define 3 lists to save 3 kinds of pixels: "skin", "grass" and "sky"
    skin = []
    grass = []
    sky = []
    # Step 3: compute the H, S, L values of each pixel, and classify them
    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j, 0], img[i, j, 1], img[i, j, 2]
            # compute the H, S, L values of each pixel
            h, s, l = rgb2hsl(r, g, b)
            # threshold the S, V values
            if s > 0.1 and l > 0.2 and l < 0.8:
                # classify different pixels
                if h >= 25 and h <= 70:
                    skin.append(s)
                if h >= 95 and h <= 135:
                    grass.append(s)
                if h >= 185 and h <= 260:
                    sky.append(s)
    # Step 4: calculate averaged saturation values for "skin", "grass" and "sky" pixels
    s_average_skin = averagenum(skin)
    s_average_grass = averagenum(grass)
    s_average_sky = averagenum(sky)
    # Step 5: calculate local CNI values for "skin", "grass" and "sky" pixels
    temp = (s_average_skin - 0.76) / 0.52
    N_skin = math.exp(- 0.5 * math.pow(temp, 2))
    temp = (s_average_grass - 0.81) / 0.53
    N_grass = math.exp(- 0.5 * math.pow(temp, 2))
    temp = (s_average_sky - 0.43) / 0.22
    N_sky = math.exp(- 0.5 * math.pow(temp, 2))
    # Step 6: calculate global CNI value
    if (len(skin) + len(grass) + len(sky)) == 0:
        N = 0
    else:
        N = (len(skin) * N_skin + len(grass) * N_grass + len(sky) * N_sky) / (len(skin) + len(grass) + len(sky))
    return N

# Compute the color colorfulness index (CCI) for one image
def SingleImageCCI(path):
    # Step 1: read the image and get rg and yb representation
    img = io.imread(path).astype(np.float64)
    rg = img[:, :, 0] - img[:, :, 1]
    yb = 0.5 * img[:, :, 0] + 0.5 * img[:, :, 1] - img[:, :, 2]
    # Step 2: compute the mean and standard variation of rg and yb
    rg_mean = np.mean(rg)
    rg_std = np.std(rg)
    yb_mean = np.mean(yb)
    yb_std = np.std(yb)
    # Step 3: compute M
    mean = np.sqrt(rg_mean * rg_mean + yb_mean * yb_mean)
    std = np.sqrt(rg_std * rg_std + yb_std * yb_std)
    M = 0.3 * mean + std
    # Step 4 (optional): judge M in range [16, 20] or not
    if M >= 16 and M <= 20:
        M_determine = 1
    else:
        M_determine = 0
    return M, M_determine
