import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import glob
import os
import matplotlib.pyplot as plt

# 设置包含 TensorBoard 日志文件的文件夹路径
log_dir = 'C:\\Users\\苏俊\\Desktop\\训练数据\\filekeep\\logs'

# 使用 glob 查找所有的 TensorBoard 日志文件
event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))

# 打印找到的所有日志文件
print("Found event files:", event_files)

values = {}  # 用于保存每个标签的所有 simple_value

# 遍历找到的所有日志文件
for event_file in event_files:
    print(f"Reading file: {event_file}")
    for e in summary_iterator(event_file):
        for v in e.summary.value:
            if v.HasField('simple_value'):
                print(f"  Tag: {v.tag}, Simple value: {v.simple_value}")
                # 将 simple_value 添加到对应标签的列表中
                if v.tag not in values:
                    values[v.tag] = []
                values[v.tag].append(v.simple_value)

# 为每个标签创建一个折线图
for tag, vals in values.items():
    plt.figure()  # 创建一个新的图形
    plt.plot(vals)
    plt.title(tag)
    plt.xlabel('Time step')
    plt.ylabel('Value')
plt.show()