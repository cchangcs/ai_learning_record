from imageai.Prediction import ImagePrediction
import os

# 获取当前路径
execution_path = os.getcwd()
# 初始化预测器
predictor = ImagePrediction()
# 设置预测器的网络类型为resnet
predictor.setModelTypeAsResNet()
# 导入模型权值文件
predictor.setModelPath(os.path.join(execution_path, 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
# 加载模型
predictor.loadModel()
# 对图片进行测试并输出测试结果
predictions, probabilities = predictor.predictImage(os.path.join(execution_path, 'test.jpg'), result_count=5)
# 输出预测到的对象及相应的置信度
for prediction, probability in zip(predictions, probabilities):
    print('name:' + prediction + "  " + 'probability:' + probability)
