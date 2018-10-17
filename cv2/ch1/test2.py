import numpy as np
import cv2


print(cv2.__version__)

# img = cv2.imread('messi5.jpg', cv2.IMREAD_COLOR)  # 读入一副彩色图像，图像的透明度会被忽略，默认参数
img = cv2.imread('messi5.jpg', cv2.WINDOW_AUTOSIZE)  # 自动调整
img = cv2.resize(img, (640, 480))


rows, cols, ch = img.shape
print('行/高', rows, '列/宽', cols, '通道', ch)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # 可以调整窗口的大小


cv2.imshow('image', img)

cv2.waitKey(delay=0)

cv2.destroyAllWindows()