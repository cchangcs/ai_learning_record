# encoding:utf-8
import cv2
import numpy as np


def nothing(x):
    pass

# 当鼠标按下时变为True
drawing = False
# 如果 mode 为 true 绘制矩形。按下 "m" 编制绘制曲线，mode = True
ix, iy = -1, -1

'''
cv2.getTrackbarPos() 函数的第一个参数是滑动条的名字
第二个参数是滑动条被放置窗口的名字
第三个参数是滑动条的默认位置
第四个参数是滑动条的最大值
第五个参数是回调函数，每次滑动条的滑动都会调用回调函数
回调函数通常都会含有一个默认参数，就是滑动条的位置
'''


# 创建回调函数
def draw_circle(event, x, y, flags, param):
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    color = (b, g, r)

    global ix, iy, drawing, mode
    # 当按下左键是返回其实位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    # 当鼠标左键按下并移动是绘制图形，event 可以查看移动，flag查看是否按下
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing is True:
            if mode is True:
                cv2.rectangle(img, (ix, iy), (x, y), color, -1)
            else:
                # 绘制圆圈，小圆点连在一起就成了线，3表示画笔的粗细
                cv2.circle(img, (x, y), 3, color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

img = np.zeros((512, 512, 3), np.uint8)
mode = False

cv2.namedWindow('image')
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1)
    if k == ord('m'):
        mode = not mode
    elif k == ord('q'):
        break
