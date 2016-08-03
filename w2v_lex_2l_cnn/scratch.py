import tensorflow as tf
import numpy as np


pooled_outputs = []

# pooled_outputs.append()

# num_filters_total = 5+5
# self.h_pool = tf.concat(3, pooled_outputs)
# self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

t = [1,2,3,4,5,6,7,8,9]
hello = tf.reshape(t, [3, 3])
sess = tf.Session()
hello  = sess.run(hello)
print hello


t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
c1 = tf.concat(0, [t1, t2])
c2 = tf.concat(1, [t1, t2])
c3 = tf.concat(1, [t1, t2])

sc1, sc2, sc3 = sess.run([c1,c2,c3])
i1 = np.zeros((1,1,1,6))
i2 = np.zeros((1,1,1,6))
i1[0,0,0,0] = 00
i1[0,0,0,1] = 01
i1[0,0,0,2] = 02
i1[0,0,0,3] = 03
i1[0,0,0,4] = 04
i1[0,0,0,5] = 05

i2[0,0,0,0] = 10
i2[0,0,0,1] = 11
i2[0,0,0,2] = 12
i2[0,0,0,3] = 13
i2[0,0,0,4] = 14
i2[0,0,0,5] = 15

# i1 = np.zeros((1,1,5))
# i2 = np.zeros((1,1,5))
# i1[0,0,0] = 00
# i1[0,0,1] = 01
# i1[0,0,2] = 02
# i1[0,0,3] = 03
# i1[0,0,4] = 04
#
# i2[0,0,0] = 10
# i2[0,0,1] = 11
# i2[0,0,2] = 12
# i2[0,0,3] = 13
# i2[0,0,4] = 14



inputs = []
inputs.append(i1)
inputs.append(i2)


h_pool = tf.concat(3, inputs)
h_pool_flat = tf.reshape(h_pool, [-1, 6])

h_pool_p = sess.run(h_pool)
h_pool_flat_p = sess.run(h_pool_flat)


x = tf.Variable(0)
assign_op = x.assign(len(inputs))
kk = sess.run(assign_op)  # or `assign_op.op.run()`
print kk


kk = [[1,2,], [3,4], [5,6]]
mylist = [ kk[i] for i in [2,1,0]]
print mylist


# sha = 'fff = %d' % len(inputs)
# sha = tf.shape(inputs[0])

h_slice = tf.slice(inputs[1], [0,0,0,0], [1,1,1,5])

h_slice_p = sess.run(h_slice)
# sha_p = sess.run(sha)
# sess.run(b)
print h_slice_p

