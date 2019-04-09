import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
from random import shuffle
import sys
import time


weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1
reduction_ratio = 4

# test_iteration = 10

total_epochs = 100

#Training batch is Size -2
batchSize = 10
iterNum = int((500/batchSize)//4)

values = [[1.0, 20.0, 5.0, 5.0, 3.0, 3.0, 100.0, 50.0, 1.0]]
values = np.repeat(values, 8, axis=0)

globTrainIndex = 0
globTestIndex = 0

def listIter(listy_x,listy_y, stringy):
    global globTrainIndex, globTestIndex
    output_x, output_y = [], []
    if stringy == "train":
        try:
            output_x = listy_x[globTrainIndex]
            output_y = listy_y[globTrainIndex]
            globTrainIndex+=1

        except IndexError:
            globTrainIndex = 0

        except Exception as e:
            print(str(e))

    else:
        try:
            output_x = listy_x[globTestIndex]
            output_y = listy_y[globTestIndex]
            globTestIndex += 1

        except IndexError:
            globTestIndex = 0
            output_x = listy_x[globTestIndex]
            output_y = listy_y[globTestIndex]

        except Exception as e:
            print(str(e))

    return output_x, output_y


def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        network = Relu(network)
        return network

def Fully_connected(x, units=9, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Avg_pooling(x, pool_size=[3,3], stride=1, padding='SAME') :
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Dropout(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Evaluate(sess):
    global test_x, test_y, fileNum, iterNum
    test_acc = 0.0
    test_loss = 0.0
    for y in range(iterNum):
        t_x, t_y = listIter(test_x, test_y, "testing")
        test_feed_dict = {
        x: t_x,
        label: t_y,
        learning_rate: epoch_learning_rate,
        training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    return test_acc, test_loss

class SE_Inception_v4():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_SEnet(x)

    def Stem(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=32, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_conv1')
            x = conv_layer(x, filter=32, kernel=[3,3], padding='VALID', layer_name=scope+'_conv2')
            block_1 = conv_layer(x, filter=64, kernel=[3,3], layer_name=scope+'_conv3')

            split_max_x = Max_pooling(block_1)
            split_conv_x = conv_layer(block_1, filter=96, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')
            x = Concatenation([split_max_x,split_conv_x])

            split_conv_x1 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv3')

            split_conv_x2 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv4')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[7,1], layer_name=scope+'_split_conv5')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[1,7], layer_name=scope+'_split_conv6')
            split_conv_x2 = conv_layer(split_conv_x2, filter=96, kernel=[3,3], padding='VALID', layer_name=scope+'_split_conv7')

            x = Concatenation([split_conv_x1,split_conv_x2])

            split_conv_x = conv_layer(x, filter=192, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv8')
            split_max_x = Max_pooling(x)

            x = Concatenation([split_conv_x, split_max_x])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_A(self, x, scope):
        with tf.name_scope(scope) :
            split_conv_x1 = Avg_pooling(x)
            split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=96, kernel=[1,1], layer_name=scope+'_split_conv2')

            split_conv_x3 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(split_conv_x3, filter=96, kernel=[3,3], layer_name=scope+'_split_conv4')

            split_conv_x4 = conv_layer(x, filter=64, kernel=[1,1], layer_name=scope+'_split_conv5')
            split_conv_x4 = conv_layer(split_conv_x4, filter=96, kernel=[3,3], layer_name=scope+'_split_conv6')
            split_conv_x4 = conv_layer(split_conv_x4, filter=96, kernel=[3,3], layer_name=scope+'_split_conv7')

            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_B(self, x, scope):
        with tf.name_scope(scope) :
            init = x

            split_conv_x1 = Avg_pooling(x)
            split_conv_x1 = conv_layer(split_conv_x1, filter=128, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_split_conv2')

            split_conv_x3 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x3 = conv_layer(split_conv_x3, filter=224, kernel=[1,7], layer_name=scope+'_split_conv4')
            split_conv_x3 = conv_layer(split_conv_x3, filter=256, kernel=[1,7], layer_name=scope+'_split_conv5')

            split_conv_x4 = conv_layer(x, filter=192, kernel=[1,1], layer_name=scope+'_split_conv6')
            split_conv_x4 = conv_layer(split_conv_x4, filter=192, kernel=[1,7], layer_name=scope+'_split_conv7')
            split_conv_x4 = conv_layer(split_conv_x4, filter=224, kernel=[7,1], layer_name=scope+'_split_conv8')
            split_conv_x4 = conv_layer(split_conv_x4, filter=224, kernel=[1,7], layer_name=scope+'_split_conv9')
            split_conv_x4 = conv_layer(split_conv_x4, filter=256, kernel=[7,1], layer_name=scope+'_split_connv10')

            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3, split_conv_x4])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Inception_C(self, x, scope):
        with tf.name_scope(scope) :
            split_conv_x1 = Avg_pooling(x)
            split_conv_x1 = conv_layer(split_conv_x1, filter=256, kernel=[1,1], layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv2')

            split_conv_x3 = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x3_1 = conv_layer(split_conv_x3, filter=256, kernel=[1,3], layer_name=scope+'_split_conv4')
            split_conv_x3_2 = conv_layer(split_conv_x3, filter=256, kernel=[3,1], layer_name=scope+'_split_conv5')

            split_conv_x4 = conv_layer(x, filter=384, kernel=[1,1], layer_name=scope+'_split_conv6')
            split_conv_x4 = conv_layer(split_conv_x4, filter=448, kernel=[1,3], layer_name=scope+'_split_conv7')
            split_conv_x4 = conv_layer(split_conv_x4, filter=512, kernel=[3,1], layer_name=scope+'_split_conv8')
            split_conv_x4_1 = conv_layer(split_conv_x4, filter=256, kernel=[3,1], layer_name=scope+'_split_conv9')
            split_conv_x4_2 = conv_layer(split_conv_x4, filter=256, kernel=[1,3], layer_name=scope+'_split_conv10')

            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3_1, split_conv_x3_2, split_conv_x4_1, split_conv_x4_2])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Reduction_A(self, x, scope):
        with tf.name_scope(scope):
            k = 256
            l = 256
            m = 384
            n = 384

            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=n, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv1')

            split_conv_x2 = conv_layer(x, filter=k, kernel=[1,1], layer_name=scope+'_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=l, kernel=[3,3], layer_name=scope+'_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=m, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Reduction_B(self, x, scope):
        with tf.name_scope(scope) :
            split_max_x = Max_pooling(x)

            split_conv_x1 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv1')
            split_conv_x1 = conv_layer(split_conv_x1, filter=384, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv2')

            split_conv_x2 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=288, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv4')

            split_conv_x3 = conv_layer(x, filter=256, kernel=[1,1], layer_name=scope+'_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=288, kernel=[3,3], layer_name=scope+'_split_conv6')
            split_conv_x3 = conv_layer(split_conv_x3, filter=320, kernel=[3,3], stride=2, padding='VALID', layer_name=scope+'_split_conv7')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3])

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :
            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])

            scale = input_x * excitation

            return scale

    def Build_SEnet(self, input_x):
        input_x = tf.pad(input_x, [[0, 0], [32, 32], [32, 32], [0, 0]])
        # size 32 -> 96
        # only cifar10 architecture

        x = self.Stem(input_x, scope='stem')

        for i in range(4) :
            x = self.Inception_A(x, scope='Inception_A'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A'+str(i))

        x = self.Reduction_A(x, scope='Reduction_A')

        for i in range(7)  :
            x = self.Inception_B(x, scope='Inception_B'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B'+str(i))

        x = self.Reduction_B(x, scope='Reduction_B')

        for i in range(3) :
            x = self.Inception_C(x, scope='Inception_C'+str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C'+str(i))

        x = Global_Average_Pooling(x)
        x = Dropout(x, rate=0.2, training=self.training)
        x = flatten(x)

        x = Fully_connected(x, layer_name='final_fully_connected')
        return x


# train_x, train_y, test_x, test_y = file_iter.iterator(480, 270, batchSize)

data_order = [i for i in range(1, fileNum + 1)]
shuffle(data_order)

x = tf.placeholder(tf.float32, shape=[None, 480, 270, 3])
label = tf.placeholder(tf.float32, shape=[None, 9])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = SE_Inception_v4(x, training=training_flag).model
shapy = logits.get_shape()
logWeight = tf.divide(logits, values)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

training, training_update = tf.metrics.accuracy(labels=tf.argmax(label, 1), predictions=tf.argmax(logits, 1), name="my_training")
test, test_update = tf.metrics.accuracy(labels=tf.argmax(label, 1), predictions=tf.argmax(logits, 1), name="my_testing")

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# saver = tf.train.import_meta_graph("./Model/SE_Inception_v4.1100.ckpt.meta")

with tf.Session() as sess:
    # ckpt = tf.train.get_checkpoint_state("./model")
    ckpt = False
    # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    if ckpt:
        # saver.restore(sess, ckpt.model_checkpoint_path)
        #saver.restore(sess, "/Model")
        pass
    else:
        sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("./logs", sess.graph)

    epoch_learning_rate = init_learning_rate
    for epoch in range(total_epochs): #, total_epochs+1
        print(epoch)
        if epoch%30 == 0:
            epoch_learning_rate = epoch_learning_rate/10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0


        #Rewrite to iterate through the list
        for i in data_order:
            globTrainIndex = 0
            globTestIndex = 0
            train_x, train_y, test_x, test_y = [None]*4

            for _ in range(int(len(train_x))):
                trainX, trainY = listIter(train_x, train_y, "train")
                train_feed_dict = {
                x:trainX,
                label: trainY,
                learning_rate:epoch_learning_rate,
                training_flag: True
                }
                x, y, _, batch_loss = sess.run([logits, label, train, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                print(x, y)
                time.sleep(5)



            test_batch_acc, test_batch_loss = Evaluate(sess)

            test_loss += test_batch_loss
            test_acc += test_batch_acc

        train_loss /= 8500
        train_acc /= 8500
        # test_acc /= (int(fileNum*iterNum))
        # test_loss /= (int(fileNum*iterNum))


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
            #saver.save()