import tensorflow as tf


# 3x3, filter 64,  2
# 3x3, filter 128, 2
# 3x3, filter 256, 4
# 3x3, filter 512, 4
# 3x3, filter 512, 4





#809000, 2

#442, 5

for a in range(len(data)/50 - 900):
    listy.append(data[x:x+900])

listy = [featureExtraction(listy[l]) for l in range(len(listy))]


#fc 4096, 4096, outputSize
# also maxpool with stride=2 after each set of convs

class ResNet:
    def __init__(self, width, height, channel, batch, model):
        self.width = width
        self.height = height
        self.channel = channel
        self.batch = batch
        self.model = model

    def modelCreation(self, X):
        conv1 = self.conv([3,3], 64, 2, prev=X, name="conv1")
        pool1 = self.pool(conv1, 1, 2, name="pool1")
        conv2 = self.conv([3,3], 128, 2, prev=pool1, name="conv2")
        pool2 = self.pool(conv2, 1, 2, name="pool2")
        conv3 = self.conv([3,3], 512, 4, prev=pool2, name="conv3")
        pool3 = self.pool(conv3, 1, 2, name="pool3")
        conv4 = self.conv([3, 3], 512, 4, prev=pool3, name="conv4")
        pool4 = self.pool(conv4, 1, 2, name="pool4")

        if self.model == "resnet":
            return self.finishNet(pool4)
        else:
            return self.conv(kernelSize=[3,3], filter=10, repetitions=1, prev=pool4, name="output")

    def conv(self, kernelSize, filter, repetitions, prev, name):
        listy = [prev]

        for x in range(repetitions):
            listy.append(tf.layers.conv2d(inputs=listy[-1], kernel_size=kernelSize, filters=filter, name="{}{}".format(name, x)))
        return listy[-1]

    def pool(self, prev, size, stride, name):
        return tf.layers.max_pooling2d(inputs=prev, pool_size=size, strides=stride, name=name)

    def finishNet(self, prev):
        fully1 = self.fc(prev, unit=4096, name="fully1")
        fully2 = self.fc(fully1, unit=4096, name="fully2")
        fully3 = self.fc(fully2, unit=10, name="fully3")
        pass

    def fc(self, prev, unit, name):
        return tf.layers.dense(inputs=prev, units=unit, name=name)