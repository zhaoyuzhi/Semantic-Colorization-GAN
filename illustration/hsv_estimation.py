import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def generate_histogram(inpath, count_1):
    cal_h = np.zeros((360,1))
    sum_var_hist = 0
    sum_var_image = 0
    count_c = 0
    for filename in os.listdir(inpath):
        filepath = os.path.join(inpath, filename)
        img = cv2.imread(filepath)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #print(hsv.shape)
        #print(hsv[:,:,0]) # 0~180
        #print(hsv[:,:,1]) # 0~255
        #print(hsv[:,:,2]) # 0~255
        hsv[:,:,0] = (hsv[:,:,0].astype(np.float32) * 2)
        hist = cv2.calcHist([hsv[:,:,0]], [0], None, [360], [0, 360])
        single_var_hist = np.var(hist)
        single_var_image = np.var(img)
        sum_var_hist += single_var_hist
        sum_var_image += single_var_image
        cal_h += hist / count_1
        count_c += 1
        print(count_c, "/", count_1)
    print(cal_h.shape)
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

    cal_h = cal_h / 256 / 256

    np.savetxt('new_global_result.csv', cal_h, delimiter=',')
    np.savetxt("new_global_result.txt", cal_h)
    np.savetxt("new_global_var.txt", var)

    plt.plot(cal_h)
    plt.xlim([0, 360])
    plt.title('Histogram')
    plt.savefig('./new_global_result.png')
    plt.show()

if __name__ == "__main__":
    inpath = '/media/ztt/6864FEA364FE72E4/zhaoyuzhi/ILSVRC2012_train_256'
    count_1 = 0
    h = 0
    w = 0
    for filename in os.listdir(inpath):
        count_1 += 1
    print(count_1)
    generate_histogram(inpath, count_1)