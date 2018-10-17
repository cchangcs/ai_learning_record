# encoding:utf-8
'''
图片比较
'''
import cv2
import pickle  # pickle提供了一个简单的持久化功能,可以将对象以文件的形式存放在磁盘上

with open('photo_mat', 'rb') as f:
    mat = pickle.load(f)

pairs = []  # 配对好的
lexX = 9  # 行
lenY = 8  # 列


def compare(i, j, img):
    for x in range(lexX):
        if x < i:
            continue
        for y in range(lenY):
            if x <= i and y < y:
                continue
            z = mat[x][y]
            # 图片相似度
            y1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            z1 = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
            '''
            第一个参数是源图像，第二个参数是模板图像，第三个参数是匹配的结果图像，第四个参数是用于指定比较的方法
            cv::TM_CCORR_NORMED：归一化的相关性匹配方法，与相关性匹配方法类似，最佳匹配位置也是在值最大处
            '''
            res = cv2.matchTemplate(z1, y1, cv2.TM_CCOEFF_NORMED)

            if res[0][0] >= 0.8:
                if i == x and j == y:
                    continue
                pairs.append((i, j, x, y, res[0][0]))
                print(i, j, x, y, res)
                cv2.moveWindow('img1', 400, 300)
                cv2.imshow('img1', img)
                cv2.moveWindow('img2', 400+img.shape[1], 300)
                cv2.imshow('img2', z)
                cv2.waitKey(500)


for i, x in enumerate(mat):
    for j, y in enumerate(x):
        compare(i, j, y)