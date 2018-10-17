# encoding:utf-8
import cv2
import numpy as np

img = cv2.imread('../data/messi5.jpg')
# 函数 cv2.pyrDown() 从一个高分辨率大尺寸的图像向上构建一个金字塔
# 尺寸变小，分辨率变低
lower_reso = cv2.pyrDown(img)
cv2.imshow("lower_reso", lower_reso)

# 从一个低分辨小尺寸的图像向下构架一个金字塔，尺寸变大但分辨率不会增加
higher_reso = cv2.pyrUp(lower_reso)
cv2.imshow('higher_reso', higher_reso)

# 使用pyrUp和pyrDown是不同的，使用pyrDow()图像的分辨率就会降低，信息就会丢失
cv2.waitKey(0)
cv2.destroyAllWindows()
