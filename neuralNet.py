import tensorflow as tf
import utils

class TwoLayerNeuralNet:



    def __init__(self, input, layer1, layer2, output):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # self.input = input
        # self.layer1 = layer1
        # self.layer2 = layer2
        # self.output = output

        self.sess = tf.InteractiveSession()

        print("INIT")

        self.x = tf.placeholder(tf.float32, shape=[input])
        self.y = tf.placeholder(tf.float32, shape=[output])

        g=tf.reshape(self.x, [input,1])
        self.x_t=tf.transpose(g)
        self.W_fc1 = weight_variable([input, layer1])
        self.b_fc1 = bias_variable([layer1])
        self.h_fc1 = tf.nn.sigmoid(tf.matmul(self.x_t, self.W_fc1) + self.b_fc1)

        self.W_fc2 = weight_variable([layer1, layer2])
        self.b_fc2 = bias_variable([layer2])
        self.h_fc2 = tf.nn.sigmoid(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)

        self.W_fc3 = weight_variable([layer2, output])
        self.b_fc3 = bias_variable([output])

        q_values = tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3

        self.action  = tf.arg_max(q_values, 1)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=q_values, labels=self.y))
        self.saver = tf.train.Saver()
        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.cost)

        self.sess.run(tf.global_variables_initializer())


    def getAction(self, state):
        a = self.sess.run([self.action], feed_dict={self.x:state})
        a=int(a[0])
        return a

    def learn(self, train_X, train_Y):
        # print("TRAINX",train_X)
        for i in range(len(train_X)):
            gg = self.sess.run([self.train_step], feed_dict={self.y: train_Y[i], self.x: train_X[i]})
            # print("SUmmary",summary)

        # train_writer.add_summary(summary, i)

    def saveState(self, model_name):
        with self.sess.as_default():
            # Save the variables to disk.
            utils.createFolderIfNotExist("./checkpoints/" + str(model_name))
            save_path = self.saver.save(self.sess, "./checkpoints/" + str(model_name) + "/model.ckpt")
            print("Model saved in file: %s" % save_path)

    def loadState(self, model_name):
        with self.sess.as_default():
            # Restore variables from disk.
            self.saver.restore(self.sess, "./checkpoints/" + str(model_name) + "/model.ckpt")
            print("Model restored from file %s." % model_name)

    def predict(self, testx):
        prediction = self.sess.run([self.action], feed_dict={ self.x: testx})
        return prediction[0]

    def resetGraph(self):
        self.sess.run(tf.global_variables_initializer())
    # def test(train_X,train_Y):
    #     # Test model
    #     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(q_values, 1))
    #     # Calculate accuracy
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #     acc = accuracy.eval({x: train_X, y: train_Y})
    #     print("Accuracy",acc)