import tensorflow as tf


class Operators:
    def __init__(self):
        pass

    def batchNorm(self, X, bool):
        return tf.layers.batch_normalization(X, training=bool)

    def Relu(self, X):
        return tf.nn.relu(X)

    def Maxpool(self, X, pool, stride):
        return tf.layers.max_pooling2d(X, pool_size=pool, strides=stride)

    def Averagepool(self, X, pool, stride):
        return tf.layers.average_pooling2d(X, pool_size=pool, strides=stride)

    def globalAveragepool(self, X):
        return tf.keras.layers.GlobalAveragePooling2D()(X)

    def dense(self, X, unit, activ):
        return tf.layers.dense(X, units=unit, activation=activ)

    def conv2d(self, X, filter, kernelSize, stride):
        return tf.layers.conv2d(X, filters=filter, kernel_size=kernelSize, strides=stride)

class dense:
    def __init__(self, name, growthRate, outputSize):
        self.o = Operators()
        self.num = 0
        self.outputSize = outputSize
        denseList = {"dense121": [6, 12, 24, 16], "dense169": [6, 12, 32, 32], "dense201":[6, 12, 48, 32], "dense264": [6, 12, 64, 48]}
        self.denseRepeats = denseList[name]
        self.growthRate = growthRate
        self.filterSize = 2*self.growthRate

    def denseBlock(self, X):
        concatList = [X]
        for y in range(self.denseRepeats[self.num]):
            if y == 0:
                conv = self.o.conv2d(X, filter=self.filterSize, kernelSize=[1, 1], stride=[1, 1])
                X = self.o.conv2d(conv, filter=self.filterSize, kernelSize=[3, 3], stride=[1, 1])
                concatList.append(X)
            else:
                conv = self.o.conv2d(X, filter=self.filterSize, kernelSize=[1, 1], stride=[1, 1])
                X = self.o.conv2d(conv, filter=self.filterSize, kernelSize=[3, 3], stride=[1, 1])
                for z in range(len(concatList)-1):
                    X = tf.concat(X, concatList[z], axis=-1)
                concatList.append(X)
        self.filterSize += self.growthRate
        self.num += 1
        return X


    def transitionBlock(self, x):
        conv = self.o.conv2d(x, filter=self.filterSize, kernelSize=[1,1], stride=[2,2])
        return self.o.Averagepool(conv, pool=[2,2], stride=[2,2])

    def startBlock(self, X):
        conv = self.o.conv2d(X, filter=self.filterSize, kernelSize=[7, 7], stride=[2, 2])
        return self.o.Maxpool(conv, pool=[3, 3], stride=[2, 2])

    def outBlock(self, X):
        pool = self.o.Averagepool(X, pool=[7, 7], stride=[1, 1])
        return self.o.dense(pool, self.outputSize, activ=tf.nn.softmax)

    def concat(self, X):
        pass

    def buildModel(self, X):
        start = self.startBlock(X)
        d1 = self.denseBlock(start)
        t1 = self.transitionBlock(d1)
        o1 = tf.concat(X, t1, axis=-1)
        d2 = self.denseBlock(o1)
        t2 = self.transitionBlock(d2)
        o2 = tf.concat(o1, t2, axis=-1)
        d3 = self.denseBlock(o2)
        t3 = self.transitionBlock(d3)
        o3 = tf.concat(o2, t3, axis=-1)
        d4 = self.denseBlock(o3)
        o4 = tf.concat(o3, d4, axis=-1)
        return self.outBlock(o4)








