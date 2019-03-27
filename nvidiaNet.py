import tensorflow as tf
import numpy as np
import tflearn
import time
import dataamount as da
from random import shuffle
import file_iter

output = 9
momentum = 0.9
width = 480
height = 270
channels = 3

total_epochs = 100

fileNum = da.fileNum()
#Training batch is Size -2
batchSize = 10
iterNum = int((500/batchSize)//4)

init_learning_rate = 0.1

def Evaluate(sess, test_feed_dict):
    loss_, _ = sess.run([cost, test_update], feed_dict=test_feed_dict)
    return loss_

def convlayers(norm):
    # Convolutional layers

    conv1 = tf.layers.conv2d(inputs=norm, filters=24, kernel_size=[5, 5], strides=[2, 2], name="Convolution_1",
                             data_format="channels_first")  # ?,24,31,98

    conv2 = tf.layers.conv2d(inputs=conv1, filters=36, kernel_size=[5, 5], strides=[2, 2], name="Convolution_2",
                             data_format="channels_first")  # ?,36,14,47

    conv3 = tf.layers.conv2d(inputs=conv2, filters=48, kernel_size=[5, 5], strides=[2, 2], name="Convolution_3",
                             data_format="channels_first")  # ?,48,5,22

    conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[3, 3], strides=1, name="Convolution_4",
                             data_format="channels_first")  # ?,64,3,20

    conv5 = tf.layers.conv2d(inputs=conv4, filters=64, kernel_size=[3, 3], strides=1, name="Convolution_5",
                             data_format="channels_first")  # ?,64,1,18
    return conv5

def fullycon(flattener):
    # Lastly use fully-connected layers
    full1 = tf.layers.dense(flattener, units=96, activation=tf.nn.sigmoid,
                            name="Dense_layer_1")  # ?,96
    full2 = tf.layers.dense(full1, units=48, activation=tf.nn.sigmoid,
                            name="Fully_connected_layer_2")  # ?,48
    full3 = tf.layers.dense(full2, units=output, activation=tf.nn.sigmoid, name="Fully_connected_layer_3")  # ?,9
    return full3

def model(network):
    # scope = "Model_"
    norm = tf.layers.batch_normalization(inputs=network, name=("Normalize")) #3, 66, 200
    conv5 = convlayers(norm)

    #Flatten to size batch_size, 1152
    flattener = tf.layers.flatten(conv5)    #?,1164

    full3 = fullycon(flattener)

    return full3

learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")

network = tf.placeholder(dtype=tf.float32, shape=[None, channels, height, width], name="Input")
label = tf.placeholder(dtype=tf.float32, shape=[None, output])
prediction = model(network)

#Creating the optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=prediction))
optimiser = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimiser.minimize(cost)

#Local variables
training, training_update = tf.metrics.accuracy(labels=tf.argmax(label, 1), predictions=tf.argmax(prediction, 1), name="my_training")
test, test_update = tf.metrics.accuracy(labels=tf.argmax(label, 1), predictions=tf.argmax(prediction, 1), name="my_testing")

running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
running_vars_initializer = tf.variables_initializer(var_list=running_vars)
training_flag = tf.placeholder(tf.bool)

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), running_vars_initializer)

sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()

data_order = [i for i in range(1, fileNum + 1)]
shuffle(data_order)


with tf.Session() as sess:
    # ckpt = tf.train.get_checkpoint_state("./model")
    ckpt = True
    # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    if ckpt:
        # saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, "/Model")
    else:
        sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("./logs", sess.graph)

    epoch_learning_rate = init_learning_rate
    for epoch in range(1): #, total_epochs+1
        print(epoch)
        if epoch%30 == 0:
            epoch_learning_rate = epoch_learning_rate/10

        pre_index = 0
        train_loss = 0.0
        test_loss = 0.0


        #Rewrite to iterate through the list
        for i in data_order:
            globTrainIndex = 0
            globTestIndex = 0
            train_x, train_y, test_x, test_y = file_iter.lessMemIter(i, 480, 270, batchSize)

            train_feed_dict = {network:train_x, label:train_y, learning_rate: init_learning_rate, training_flag: True}
            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)

            train_loss += batch_loss

            test_feed_dict = {
                network: test_x,
                label: test_y,
                learning_rate: init_learning_rate,
                training_flag: False
            }

            test_batch_loss = Evaluate(sess, test_feed_dict)

            test_loss += test_batch_loss

        train_loss /= len(data_order)
        test_loss /= len(data_order)
        train_acc, test_acc = sess.run([training, test])


        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

        test_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                    tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])


        summary_writer.add_summary(summary=train_summary, global_step=epoch)
        summary_writer.add_summary(summary=test_summary, global_step=epoch)
        summary_writer.flush()

        line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
            epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
        print(line)

        with open('logs.txt', 'a') as f:
            f.write(line)

        saver.save(sess=sess, save_path='./model/SE_Inception_v4{}.ckpt'.format(epoch))

