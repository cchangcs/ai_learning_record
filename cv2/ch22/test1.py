# encoding:utf-8
'''
斑点检测SimpleBlodDetector()
斑点检测：默认检测黑色点，如果要检测白色的点需要设置bycolor为true，并且color数值为255
斑点通常是指与周围有着颜色和灰度差别的区域，在实际的图中，往往存在着大量这样的斑点，如一棵树是一个斑点，一块草地是一个斑点。
由于斑点代表的是一个区域，相比单纯的角点，它的稳定性更好，抗噪声能力更强，所以它在图像配准上扮演着重要的角色。
同时有时图像中的斑点也是我们关心的区域，比如在医学与生物领域，我们需要从一些X光照片或细胞显微照片中提取一些具有特殊意义的斑点的位置或数量
'''
import cv2
import numpy as np

im = cv2.imread('blob.jpg', cv2.IMREAD_GRAYSCALE)

detector = cv2.SimpleBlobDetector_create()

keypoints = detector.detect(im)

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Keypoints", im_with_keypoints)

cv2.waitKey(0)
cv2.destroyAllWindows()
