import tensorflow as tf

# print(tf.__version__)

# 1d tensor
# x = tf.constant(1)
x = tf.constant(1,shape=(1,2),dtype=tf.float32)
# print(x)



# 2d tensor
x = tf.constant([[1,2,3],[4,5,6]],dtype=tf.float32)
# print(x)

y = tf.ones((3,3))
z = tf.zeros((3,3))
i = tf.eye(3)
# print(y)
# print(z)
# print(i)



# math
x = tf.constant([1,2,3])
y = tf.constant([9,5,7])

# z = tf.add(x,y)
# same
z = x+y
# print(z)

# same for subtract or division or multiply

z = tf.tensordot(x,y,axes=1)
# print(z)

z = x**5
# print(z)


# Indexing
# same as py
x = tf.constant([1,2,4,5,6,7,9,4])
# print(x[1:5])



# reshape
a = tf.range(9)
print(a)

a = tf.reshape(a,(3,3))
print(a)

a = tf.transpose(a, perm=[1,0])
print(a)


