import numpy as np
import cv2, sys


image_path = 'messi5.jpg'

try:
    f = open(image_path)
except Exception as e:
    print(e)
    sys.exit(0)

img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 包括图像的 alpha 通道
temp = img.copy()

title = image_path.split('/')[-1] + f' {img.shape}'

gray = False

while True:
    cv2.imshow(title, temp)
    k = cv2.waitKey(10)
    if k == 27 or k == ord('g'):
        break
    # 分辨率太大，需要缩放
    if k == ord('g'):
        if gray is False:
            temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            greay = True
        else:
            temp = img.copy()
            gray = False
cv2.destroyAllWindows()