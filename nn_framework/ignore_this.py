# def dot_grads(var, a, b, name):
#   with tf.name_scope(name):
#     fill = tf.fill((n_h, m), 1.)
#     with tf.name_scope('matmul_grad'):
#       da_target = tf.matmul(fill, b, transpose_b=True)
#       db_target = tf.matmul(fill, a, transpose_a=True)

#     [da] = tf.gradients(a, [var], da_target, name='da')
#     [db] = tf.gradients(b, [var], tf.transpose(db_target), name='db')

#     if da is None:
#       return db
#     elif db is None:
#       return da
#     else:
#       return tf.add_n([da, db])
