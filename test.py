import tensorflow as tf

input1 = tf.convert_to_tensor([[1],[1],[1],[1],[1],[1],[1],[1],[1]], dtype=tf.float32)
input2 = tf.convert_to_tensor([[1],[1],[1],[1],[1],[1],[1],[1],[1]], dtype=tf.float32)
input3 = tf.convert_to_tensor([[0],[1],[1],[1],[1],[1],[1],[1],[1]], dtype=tf.float32)

print(tf.equal(input1,input2))
print(tf.equal(input1,input3))

