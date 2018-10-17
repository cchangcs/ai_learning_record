# encoding:utf-8
'''
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None)
img： 输入图像
mask：掩码图像，用来确定哪些区域是背景、前景，哪些可能是前景/背景，可以置为 cv2.GC_BGD,cv2.GC_FGD,cv2.GC_PR_BGD,cv2.GC_PR_FGD  或者直接填入 0,1,2,3
rect：包括前景的矩形，格式为(x,y,w,h)
bdgModel， fgdModel：算法内使用的数组，只需要创建两个大小为（1，65），数据类型为 np.float64 的数组
iterCount：算法的迭代次数
mode 可以置为 cv2.GC_INIT_WITH_RECT或cv2.GC_INIT_WITH_MASK，也可以联合使用，是用来确定我们修改的方式使用矩形形式还是掩码形式
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../data/messi5.jpg')

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, 450, 290)

# 函数的返回值是更新的 mask，bgdModel， fgdModel
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]

plt.imshow(img), plt.colorbar(), plt.show()
