# -*-coding:utf8-*-#
'''
模板匹配是用来在一副大图中寻找模板图像位置的方法。OpenCV为我们提供了函数cv2.matchTemplate()
和 2D 卷积一样，它也是用模板图像在输入图像大图上滑动，并在每一个位置对模板图像和输入图像的子区域进行比较
OpenCV提供了集中不同的比较方法
返回结果是一个灰度图像，每一个像素值显示了此区域与模板的匹配程度
如果输入图像的大小是WxH
模板的大小时wxh，输出的结果就是W-w+1 H-h+1
得到这幅图之后，就可以使用函数cv2.minMaxLoc()来找到其中的最小值和最大值的位置
第一个值为矩形左上角的点的位置
w h 就是模板矩形的宽和高
这个矩形就是找到模板区域
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../data/messi5.jpg', 0)
img2 = img.copy()
template = cv2.imread('../data/messi_face.jpg', 0)

# cv2.imshow('messi', img)
# cv2.imshow('face', template)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    '''
    exec可以用来执行储存在字符串货文件中的python语句
    例如可以在运行时生成一个包含python代码的字符串
    然后使用exec语句执行这些语句
    eval语句用来计算存储在字符串中的有效python表达式
    '''
    method = eval(meth)
    # Apply template matching
    res = cv2.matchTemplate(img, template, method)
    mn_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 使用不同的方法，对结果的解释不同
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle('method: ' + meth)
    plt.show()
