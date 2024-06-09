import numpy as np
import tensorflow as tf
import os

preprocessed_images = np.random.rand(1, 3, 224, 224).astype(np.float32)

# 加载TFLite模型
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# 获取输入和输出张量的详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 设置输入张量
interpreter.set_tensor(input_details[0]['index'], preprocessed_images)

# 运行模型推断
interpreter.invoke()

# 获取输出数据
output_data = interpreter.get_tensor(output_details[0]['index'])

# 或者其他形式的处理输出数据
print(output_data) 


