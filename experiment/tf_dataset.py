import tensorflow as tf
import numpy as np

dataset = tf.data.Dataset.from_tensor_slices(np.array([1, 2, 3, 4, 5], dtype=np.int32))
for i in dataset:
    print(i.numpy())

# it = iter(dataset)
# for i in dataset:
#     print(it.next().numpy())


print(dataset.reduce(0, lambda state, value: state + value).numpy())
