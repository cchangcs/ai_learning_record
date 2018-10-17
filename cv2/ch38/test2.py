# encoding:utf-8
import cv2
import numpy as np

# 第一阶段将所有图像加载到列表中，此外，需要常规HDR的曝光时间
# 需要注意数据类型：图像应为1通道或3通道8位（np.uint8），曝光时间需要为np.float32，以秒为单位
img_fn = ['1tl.jpg', '2tr.jpg', '3bl.jpg', '4br.jpg']
img_list = [cv2.imread(fn) for fn in img_fn]
exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

# 将曝光序列合并成一个HDR图像，显示了在OpenCV中求高动态范围成像的两种算法：Debvec和Robertson，
# HDR图像的类型为float32，而不是uint8,应为它包含所有曝光图像的完整动态范围
merge_debvec = cv2.createMergeDebevec()
hdr_debvec = merge_debvec.process(img_list, times=exposure_times.copy())
merge_robertson = cv2.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# 将32位浮点HDR数据映射到范围[0..1]。实际上，在某些情况下，值可能大于1或低于0，所以注意我们以后不得不剪切数据，以避免溢出。
tonemap1 = cv2.createTonemapDurand(gamma=2.2)
res_debvec = tonemap1.process(hdr_debvec.copy())
tonemap2 = cv2.createTonemapDurand(gamma=1.3)
res_robertson = tonemap2.process(hdr_robertson.copy())

# 合并曝光图像的替代算法，不需要曝光时间。也不需要使用任何tonemap算法，因为Mertens算法已经给出了[0..1]范围内的结果。
merge_mertens = cv2.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# Convert datatype to 8-bit and save
# 为了保存或显示结果，我们需要将数据转换为[0..255]范围内的8位整数。
res_debvec_8bit = np.clip(res_debvec * 255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson * 255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')

cv2.imshow("ldr_debvec.jpg", res_debvec_8bit)
cv2.imshow("ldr_robertson.jpg", res_robertson_8bit)
cv2.imshow("fusion_mertens.jpg", res_mertens_8bit)
cv2.waitKey(0)
cv2.destroyAllWindows()
