import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[8])
y = tf.placeholder(tf.float32, shape=[2])

layer1 = 20
layer2 = 20

g=tf.reshape(x, [8,1])
x_t=tf.transpose(g)
W_fc1 = weight_variable([8, layer1])
b_fc1 = bias_variable([layer1])
h_fc1 = tf.nn.sigmoid(tf.matmul(x_t, W_fc1) + b_fc1)

W_fc2 = weight_variable([layer1, layer2])
b_fc2 = bias_variable([layer2])
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([layer2, 2])
b_fc3 = bias_variable([2])

q_values = tf.matmul(h_fc2, W_fc3) + b_fc3

action  = tf.arg_max(q_values, 1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=q_values, labels=y))
# q_a_max = tf.reduce_max(q_values)

# loss = tf.square(target - q_a_max)
with tf.name_scope('loss'):
    # loss = tf.clip_by_value(loss, 0, 1
    tf.summary.scalar('TD Error', cost)

train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)

merged = tf.summary.merge_all()
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
# train_writer = tf.summary.FileWriter('./train', sess.graph)
sess.run(tf.initialize_all_variables())

def getAction(state):
    a = sess.run([action], feed_dict={x:state})
    a=int(a[0])
    return a

def learn(train_X, train_Y):
    # print("TRAINX",train_X)
    for i in range(len(train_X)):
        gg, summary = sess.run([train_step, merged], feed_dict={y: train_Y[i], x: train_X[i]})
        # print("SUmmary",summary)

    # train_writer.add_summary(summary, i)

def predict(testx):
    prediction = sess.run([action], feed_dict={ x: testx})
    return prediction[0]
# def test(train_X,train_Y):
#     # Test model
#     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(q_values, 1))
#     # Calculate accuracy
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     acc = accuracy.eval({x: train_X, y: train_Y})
#     print("Accuracy",acc)