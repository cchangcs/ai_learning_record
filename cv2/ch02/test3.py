import numpy as np
import cv2


cap = cv2.VideoCapture("Minions_banana.mp4")

# 帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

# 总共有多少帧
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("总共", num_frames, '帧')

frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print('宽：', frame_width, '高：', frame_height)

frame_now = cap.get(cv2.CAP_PROP_POS_FRAMES)
print('当前帧数：', frame_now)

# 读取指定帧，对视频文件才有效，对摄像头无效
frame_no = 121
cap.set(1, frame_no)
ret, frame = cap.read()
cv2.imshow('frame_no' + str(frame_no), frame)

frame_now = cap.get(cv2.CAP_PROP_POS_FRAMES)
print('当前帧数：', frame_now)

while cap.isOpened():
    ret, frame = cap.read()
    frame_now = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print('当前帧数：', frame_now)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
