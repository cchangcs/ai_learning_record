# encoding:utf-8
'''
算法使用的是已经平滑后的图像
BRIEF是一种特征描述符，它不提供查找特征的方法，所以需要使用其他的特征检测器，比如SIFT和SURF等
推荐使用CenSurf 特征检测器，这种算法很快，而且BRIEF算法对CenSurf关键点的描述效果比SURF关键点的描述更好
BRIED是一种对特征点描述符运算和匹配的快速方法，这种算法可以实现很高的识别率，除非出现了平面内的大旋转。
'''
import cv2
import numpy as np


img = cv2.imread('../data/blox.jpg', 0)

# 初始化FAST检测器
star = cv2.xfeatures2d.StarDetector_create()
# 初始化BRIEF特征描述符
breif = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# 通过star找到关键点
kp = star.detect(img, None)
# 通过BRIEF计算描述子
kp, des = breif.compute(img, kp)

print(breif.descriptorSize())
print(des.shape)


