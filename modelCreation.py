import tensorflow as tf
import keras, numpy as np

#Operators used for convolutions, and such to minimize
# the amount of writing one has to do
class operators:
    def __init__(self):
        pass

    def conv(self, x, filterSize, kS, stride):
        return tf.layers.conv2d(inputs=x, filters=filterSize, kernel_size=kS, strides=stride)

    def relu(self, x):
        return tf.nn.relu(features=x)

    def seperableConv(self, x, filt):
        return tf.layers.separable_conv2d(inputs=x, filters=filt, kernel_size=[3, 3],
                                          depthwise_initializer=tf.contrib.layers.xavier_initializer(),
                                          pointwise_initializer=tf.contrib.layers.xavier_initializer(),
                                          strides=[1, 1])

    def maxPool(self, x):
        return tf.layers.max_pooling2d(inputs=x, pool_size=[3, 3], strides=[2, 2])

    def globalAvgPooling(self, x):
        return tf.keras.layers.GlobalAveragePooling2D(inputs=x)

    #Residual block (adds x, and y together with a convolution on x)
    def residual(self, x, y, filt):
        res0_0 = tf.layers.conv2d(x, filters=filt, kernel_size=[1, 1], strides=[2, 2])
        return tf.add(res0_0, y)

#Creates the Xception network
class model:
    def __init__(self):
        self.op = operators()

    #Used to automate repetative parts of the middleFlow
    def middleRepetition(self, x, filty):
        conv0 = self.relusep(x, filty)
        conv1 = self.relusep(conv0, filty)
        conv2 = self.relusep(conv1, filty)
        return tf.add(conv2, x)

    #Relu then separable convolution
    def relusep(self, x, filty):
        relu0_0 = self.op.relu(x)
        return self.op.seperableConv(relu0_0, filty)

    #Separable convolution then relu
    def seprelu(self, x, filty):
        sep0 = self.op.seperableConv(x, filty)
        return self.op.relu(sep0)

    #Used to automate repetative parts of the entryFlow
    def entryRepetition(self, x, filt):
        relu0_0 = self.op.relu(x)
        sepconv0_0 = self.op.seperableConv(relu0_0, filt)
        relu0_1 = self.op.relu(sepconv0_0)
        sepconv0_1 = self.op.seperableConv(relu0_1, filt)
        maxpool0_0 = self.op.maxPool(sepconv0_1)
        return self.op.residual(x=x, y=maxpool0_0, filt=filt)


    def entryFlow(self, X):
        conv0_0 = self.op.conv(X, filterSize=32, kS=[3,3], stride=[2,2])
        relu0_0 = self.op.relu(conv0_0)
        conv0_1 = self.op.conv(relu0_0, filterSize=64, kS=[3, 3], stride=[1, 1])
        relu0_1 = self.op.relu(conv0_1)
        sepconv0_0 = self.op.seperableConv(relu0_1, 128)
        relu0_2 = self.op.relu(sepconv0_0)
        sepconv0_1 = self.op.seperableConv(relu0_2, 128)
        maxpool0_0 = self.op.maxPool(sepconv0_1)
        res0_0 = self.op.residual(relu0_1, maxpool0_0, 128)
        res1_0 = self.entryRepetition(res0_0, 256)
        res2_0 = self.entryRepetition(res1_0, 728)
        return res2_0

    def middleFlow(self, X):
        for a in range(8):
            X = self.middleRepetition(X, 728)
        return X

    def exitFlow(self, X, outSize):
        conv0 = self.relusep(X, 728)
        conv1 = self.relusep(conv0, 1024)
        maxpool0 = self.op.maxPool(conv1)
        res0 = self.op.residual(X, maxpool0, 1024)
        conv2 = self.seprelu(res0, 1536)
        conv3 = self.seprelu(conv2, 2048)
        globavg0 = self.op.globalAvgPooling(conv3)
        # Finish of the network
        return tf.layers.dense(inputs=globavg0, units=outSize)

    #Creates the networks output (the only function which has to be run)
    def modelCreation(self, X, actionSize):
        entry = self.entryFlow(X)
        middle = self.middleFlow(entry)
        return self.exitFlow(middle, actionSize)


def modelCreation():
    return keras.applications.xception.Xception(include_top=True, weights="imagenet", classes=54)

