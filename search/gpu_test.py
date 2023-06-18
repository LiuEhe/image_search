import tensorflow as tf

# 获取可用的物理设备列表
physical_devices = tf.config.experimental.list_physical_devices()

# 打印每个设备的名称
for device in physical_devices:
    print('设备名：', device.name)