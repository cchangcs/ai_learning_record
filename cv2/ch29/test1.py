# encoding:utf-8
'''
SIFT效果很好，但是从实时处理的角度来看，这些算法的效果不是很好
一个很好的例子就是SLAM（同步定位与地图构建）
因此需要使用到快速特征检测器
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../data/blox.jpg', 0)

# 使用默认值初始化检测对象
fast = cv2.FastFeatureDetector_create()
# 找到并绘制关键点keypoints
kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

# 输出所有的默认参数
print('Threshold:', fast.getThreshold())
print('nonmaxSuppression:', fast.getNonmaxSuppression())
print('neightborhood:', fast.getType)
print('Total Keypoints with nonmaxSuppression:', len(kp))
# 使用最大值抑制的结果
cv2.imshow('fast_true', img2)

# 未使用最大值抑制的结果
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print('Total Keypoints without nonmaxSuppression:', len(kp))

img3 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
cv2.imshow('fast_false', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()


