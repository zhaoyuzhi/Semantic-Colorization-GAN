import cv2
import numpy as np
import os
import random
from matplotlib import pyplot as plt

def sal_patch_judge(salpath, rgbpath, count_1, option):
    cal_h = np.zeros((360, 1))
    sum_var_hist = 0
    sum_var_image = 0
    count_c = 0
    for filename in os.listdir(salpath):
        sal_file = os.path.join(salpath, filename)
        rgb_file = os.path.join(rgbpath, filename)
        #print(sal_file)
        rgb = cv2.imread(rgb_file)
        sal = cv2.imread(sal_file, cv2.IMREAD_GRAYSCALE)
        high = sal > 20
        #print(np.sum(high))
        high = (high * 255).astype(np.uint8)
        count = 0
        dflag = 0
        while count < 100 and dflag == 0:
            h, w = sal.shape[:2]
            rand_h = random.randint(0, h - h // 4)
            rand_w = random.randint(0, w - w // 4)
            rgb_patch = rgb[rand_h : rand_h + h // 4, rand_w : rand_w + w // 4, :]
            high_patch = high[rand_h : rand_h + h // 4, rand_w : rand_w + w // 4]
            if option == 1:
                if np.sum(high_patch) > 2048 * 255:
                    dflag = 1
                    hsv = cv2.cvtColor(rgb_patch, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.float32) * 2)
                    hist = cv2.calcHist([hsv[:, :, 0]], [0], None, [360], [0, 360])
                    single_var_hist = np.var(hist)
                    single_var_image = np.var(rgb_patch)
                    sum_var_hist += single_var_hist
                    sum_var_image += single_var_image
                    cal_h += hist / count_1
            elif option == 2:
                if np.sum(high_patch) < 2048 * 255:
                    dflag = 1
                    hsv = cv2.cvtColor(rgb_patch, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.float32) * 2)
                    hist = cv2.calcHist([hsv[:, :, 0]], [0], None, [360], [0, 360])
                    single_var_hist = np.var(hist)
                    single_var_image = np.var(rgb_patch)
                    sum_var_hist += single_var_hist
                    sum_var_image += single_var_image
                    cal_h += hist / count_1
            elif option == 3:
                dflag = 1
                hsv = cv2.cvtColor(rgb_patch, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0].astype(np.float32) * 2)
                hist = cv2.calcHist([hsv[:, :, 0]], [0], None, [360], [0, 360])
                single_var_hist = np.var(hist)
                single_var_image = np.var(rgb_patch)
                sum_var_hist += single_var_hist
                sum_var_image += single_var_image
                cal_h += hist / count_1
            count += 1
        count_c += 1
        print(count_c, "/", count_1)

    print("sum_var_hist", sum_var_hist)
    print("sum_var_image", sum_var_image)
    avg_var_hist = sum_var_hist / count_1
    avg_var_image = sum_var_image / count_1
    print("avg_var_hist", avg_var_hist)
    print("avg_var_image", avg_var_image)

    var = np.zeros((4, 1))
    var[0][0] = sum_var_hist
    var[1][0] = sum_var_image
    var[2][0] = avg_var_hist
    var[3][0] = avg_var_image

    cal_h = cal_h / 64 / 64
    np.savetxt('new_result_1.csv', cal_h, delimiter=',')
    np.savetxt("new_result_1.txt", cal_h)
    np.savetxt("new_var_1.txt", var)

    plt.plot(cal_h)
    plt.xlim([0, 360])
    plt.title('Histogram')
    plt.savefig('./new_result_1.png')
    plt.show()


if __name__ == "__main__":
    rgbpath ='/media/ztt/6864FEA364FE72E4/zhaoyuzhi/ILSVRC2012_train_256'
    salpath ='/media/ztt/6864FEA364FE72E4/zhaoyuzhi/ILSVRC2012_train256_saliency'
    option = 1
    # 1: high > 50%
    # 2: high < 50%
    # 3: random
    count_1 = 0
    for filename in os.listdir(rgbpath):
        count_1 += 1
    print(count_1)

    sal_patch_judge(salpath, rgbpath, count_1, option)
    