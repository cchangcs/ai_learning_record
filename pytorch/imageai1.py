from imageai.Detection import ObjectDetection
import os

# 获取当前路径
execution_path = os.getcwd()
# 初始化检测器
detector = ObjectDetection()
# 设置检测器的网络类型为resnet
detector.setModelTypeAsRetinaNet()
# 导入模型权值文件
detector.setModelPath(os.path.join(execution_path, 'resnet50_coco_best_v2.0.1.h5'))
# 加载模型
detector.loadModel()
# 对图片进行测试并输出测试结果
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, 'test.jpg'),
                                         output_image_path=os.path.join(execution_path, 'result.jpg'))
# 输出检测到的对象及相应的置信度
for object in detections:
    print('name:' + object['name'] + "  " + 'probability:' + object['percentage_probability'])