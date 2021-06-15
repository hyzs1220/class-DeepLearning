# FInal Experiment Text Sentiment Classification

## Experiment 

+ 在本次实验中，因为是对电影的评论数据进行一个文本语句分类，所以采用了TextCNN网络模型（网络实现部分参考了 https://github.com/XqFeng-Josie/TextCNN ，基于 TensorFlow 实现）
+  ![自然语言中的CNN--TextCNN（基础篇）](https://pic1.zhimg.com/v2-cc89511361127dadf0f831a67a5e3b60_1440w.jpg?source=172ae18b) 

+ 首先对本次实验的数据格式进行相应的处理

```python
import os
from tqdm import tqdm

# filepath = './train/neg/'
filepath = './train/pos/'

# filepath = './test/'
filename = os.listdir(filepath)

# filename_write = 'rt-polarity.neg'
filename_write = 'rt-polarity.pos'
with open(filename_write,'w') as f_write:
    for f_item in tqdm(filename):
        # print(f_item)
        with open(filepath + f_item, 'r', errors='ignore') as f_read:
            contents = f_read.read()
            f_write.write(contents + '\n')

```

+ 网络训练的主体部分

```python
for batch in batches:
    x_batch, y_batch = zip(*batch)
    train_step(x_batch, y_batch,train_op)
    current_step = tf.train.global_step(sess, global_step)
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
```

+ 运行网络进行训练，并在验证集上记录每一次的准确率和loss，查看网络运行效果
+ ![1623766334392](final-exp-Text Sentiment Classification\1623766334392.png)
+ ![1623766334392](final-exp-Text Sentiment Classification\Figure_1.png)

