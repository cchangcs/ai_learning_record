import os
import shutil  # 复制文件
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image

# 原始目录所在的路径
# 数据集未压缩
original_dataset_dir = 'kaggle_original_data'

# 存储较小数据集的目录
base_dir = 'cats_and_dogs_small'
os.mkdir(base_dir)

# 训练、验证、测试数据集的目录
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# 猫训练图片所在目录
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# 狗训练图片所在目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# 猫验证图片所在目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# 狗验证数据集所在目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# 猫测试数据集所在目录
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# 狗测试数据集所在目录
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# 复制最开始的1000张猫图片到 train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# 复制接下来500张猫图片到 validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

# 复制接下来500张图片到 test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# 复制最开始的1000张狗图片到 train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 复制接下来500张狗图片到 validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

# 复制接下来500张狗图片到 test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))

print('total training dog images:', len(os.listdir(train_dogs_dir)))

print('total validation cat images:', len(os.listdir(validation_cats_dir)))

print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

print('total test cat images:', len(os.listdir(test_cats_dir)))

print('total test dog images:', len(os.listdir(test_dogs_dir)))

# 搭建模型
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])


'''
数据应该在被送入我们的网络之前格式化为适当的预处理浮点张量。
目前，我们的数据作为JPEG文件位于驱动器上，因此将其放入网络的步骤大致如下：

*阅读图片文件。
*将JPEG内容解码为RBG像素网格。
*将这些转换为浮点张量。
*将像素值（0到255之间）重新缩放到[0,1]间隔（如您所知，神经网络更喜欢处理小输入值）。

  这可能看起来有点令人生畏，但幸好Keras有实用工具自动处理这些步骤。
 Keras有一个带有图像处理辅助工具的模块，位于`keras.preprocessing.image`。
 特别是，它包含类ImageDataGenerator，它允许快速设置Python生成器，
 可以自动将磁盘上的图像文件转换为批处理的预处理张量。
'''

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,  # target directory
    target_size=(150, 150),  # resize图片
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# hist = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=30,
#     validation_data=validation_generator,
#     validation_steps=50
# )
#
# model.save('cats_and_dogs_small_1.h5')
#
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
#
# plt.legend()
# plt.figure()
#
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.legend()
# plt.show()

'''
因为我们只有相对较少的训练样本（2000），过度拟合将是我们的头号问题。您已经了解了许多有助于缓解过度拟合的技术，
例如丢失和重量衰减（L2正规化）。我们现在将介绍一种新的，专门针对计算机视觉的，并且在使用深度学习模型处理图像时几乎普遍使用：*数据增强*。

###使用数据增强
＃过度拟合是由于样本太少而无法学习，导致我们无法训练能够推广到新数据的模型。
＃给定无限数据，我们的模型将暴露于手头数据分布的每个可能方面：我们永远不会过度拟合。
数据增强采用从现有训练样本生成更多训练数据的方法，通过数字“扩充”样本
随机转换的数量，产生可信的图像。目标是在训练时，我们的模型永远不会看到两次完全相同的图片。
这有助于模型暴露于数据的更多方面并更好地概括。
'''
# *`rotation_range`是一个度数（0-180）的值，是一个随机旋转图片的范围。
# *`width_shift`和`height_shift`是范围（作为总宽度或高度的一小部分），在其中随机翻译图片
# 垂直或水平。
# *`shear_range`用于随机应用剪切变换。
# *`zoom_range`用于随机缩放图片内部。
# *`horizontal_flip`用于水平地随机翻转一半图像 - 当没有水平假设时相关
# #unymmetry（例如真实世界的图片）。
# *`fill_mode`是用于填充新创建的像素的策略，可以在旋转或宽度/高度移位后出现。
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# # 查看数据增强的效果
# frames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
# # 选择一张图片来做增强
# img_path = fnames[3]
#
# # 读取图片并进行resize
# img = image.load_img(img_path, target_size=(150, 150))
#
# # 转化为Numpy数组， shape(150, 150, 3)
# x = image.img_to_array(img)
# # reshape->(1, 150, 150, 3)
# x = x.reshape(1, 150, 150, 3)
#
# i = 0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break
# plt.show()

# 使用数据增强后的数据来训练一个新的网络
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
                 input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./ 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=4,
    validation_data=validation_generator,
    validation_steps=50,
)

model.save('cats_and_dogs_small_2.h5')

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()