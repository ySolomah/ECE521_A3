import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import numpy as np
import random



rng = np.random

batch_size = 500
lr = 0.002
lam = 0.002
num_iter = 1000
num_neurons = [1000]


validTrain = 1
initializer = tf.contrib.layers.xavier_initializer()

def fcLayer(inTensor, numHidden, spatialSize=784, lamb=0):

    weights = tf.Variable(initializer([spatialSize, numHidden]), name="W" + str(spatialSize))
    bias = tf.Variable(tf.zeros([numHidden]), name="b" + str(spatialSize))
    bias = bias + 0.1
    return(tf.matmul(inTensor, weights) + bias, (lamb / 2) * tf.reduce_sum(tf.matmul(weights, weights, transpose_b=True)), weights, bias)

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]

    trainData, trainTarget = Data[:15000], Target[:15000]
    trainTargetCopy = np.copy(trainTarget)
    trainTarget = np.expand_dims(trainTarget, axis=1)
    z = np.zeros((trainTarget.shape[0], 9))
    trainTarget = np.concatenate((trainTarget, z), axis=1)

    validData, validTarget = Data[15000:16000], Target[15000:16000]
    validTargetCopy = np.copy(validTarget)
    validTarget = np.expand_dims(validTarget, axis=1)
    z = np.zeros((validTarget.shape[0], 9))
    validTarget = np.concatenate((validTarget, z), axis=1)

    testData, testTarget = Data[16000:], Target[16000:]
    testTargetCopy = np.copy(testTarget)
    testTarget = np.expand_dims(testTarget, axis=1)
    z = np.zeros((testTarget.shape[0], 9))
    testTarget = np.concatenate((testTarget, z), axis=1)
      
    for edit_array in [trainTarget, validTarget, testTarget]:
        for row in edit_array:
            idx = row[0].astype(int)
            row[0] = 0
            row[idx] = 1


    for hiddenCount, colour in zip(num_neurons, ['mo', 'bo', 'ro', 'go', 'yo']):
        epoch_array = []
        train_accuracy = []
        valid_accuracy = []
        test_accuracy = []
        train_loss = []
        valid_loss = []
        test_loss = []
        print("Learning Rate : " + str(lr))
        X = tf.placeholder("float")
        y = tf.placeholder("float")

        intermediateLayer, regularizeW1, W1, b1 = fcLayer(X, hiddenCount, lamb=lam)
        featureMap = tf.nn.relu(intermediateLayer)
        outputLayer, regularizeW2, W2, b2 = fcLayer(featureMap, 10, spatialSize=hiddenCount, lamb=lam)        
        temp4_5 = tf.nn.softmax_cross_entropy_with_logits(logits=outputLayer, labels=y)
        sigmoids = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputLayer, labels=y)
        arg_max = tf.argmax(sigmoids, dimension=1)
        softmax_acc = tf.argmax(tf.nn.softmax(outputLayer), dimension=1)
        temp5 = tf.reduce_sum(temp4_5)
        loss = (1 / batch_size) * temp5
        total_loss = loss + regularizeW1 + regularizeW2

        optim = tf.train.AdamOptimizer(lr).minimize(total_loss)

        init = tf.global_variables_initializer()

        trainDataReshapeMatt = trainData.reshape(-1, trainData.shape[1] * trainData.shape[2])

        trainDataReshaped2 = trainData.reshape([trainData.shape[0], trainData.shape[1] * trainData.shape[2]])
        trainTargetReshaped2 = trainTarget.reshape([trainTarget.shape[0], trainTarget.shape[1]])
        trainDataReshaped = trainData.reshape([-1, batch_size, trainData.shape[1] * trainData.shape[2]])
        trainTargetReshaped = trainTarget.reshape([-1, batch_size, trainTarget.shape[1]])
        testDataReshaped = testData.reshape([testData.shape[0], testData.shape[1] * testData.shape[2]])
        testTargetReshaped = testTarget.reshape([testTarget.shape[0], testTarget.shape[1]])
        validDataReshaped = validData.reshape([validData.shape[0], validData.shape[1] * validData.shape[2]])
        validTargetReshaped = validTarget.reshape([validTarget.shape[0], validTarget.shape[1]])

        saver = 0

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(int(num_iter / trainDataReshaped.shape[0])):
                new_epoch = True
                for miniBatchData, miniBatchTarget in zip(trainDataReshaped, trainTargetReshaped):
                    minibatch = random.sample(list(zip(trainDataReshapeMatt, trainTarget)), batch_size)
                    miniBatchData, miniBatchTarget = zip(*minibatch)
                    if(new_epoch):
                        if(saver == 3 and epoch > 0.99 * int(num_iter / trainDataReshaped.shape[0])):
                            tf.train.Saver().save(sess, '/Users/yassirsolomah/ECE521_A3/ECE521_A3/99save1_1')
                            saver = saver + 1
                        if(saver == 2 and epoch > 0.74 * int(num_iter / trainDataReshaped.shape[0])):
                            tf.train.Saver().save(sess, '/Users/yassirsolomah/ECE521_A3/ECE521_A3/74save1_1')
                            saver = saver + 1
                        if(saver == 1 and epoch > 0.49 * int(num_iter / trainDataReshaped.shape[0])):
                            tf.train.Saver().save(sess, '/Users/yassirsolomah/ECE521_A3/ECE521_A3/49save1_1')
                            saver = saver + 1
                        if(saver == 0 and epoch > 0.24 * int(num_iter / trainDataReshaped.shape[0])):
                            tf.train.Saver().save(sess, '/Users/yassirsolomah/ECE521_A3/ECE521_A3/24save1_1')
                            saver = saver + 1
                        new_epoch = False
                        if(epoch % 100 == 0):
                            print("\n\nEpoch : " + str(epoch) +  "\n Loss : " + str(sess.run(loss, feed_dict={X: miniBatchData, y: miniBatchTarget})) + "\n Total Loss : " + str(sess.run(total_loss, feed_dict={X: miniBatchData, y: miniBatchTarget})))
                        epoch_array.append(epoch)
                        '''
                        if(validTrain == 0):
                        	guessIn = trainDataReshaped2
                        	guessOut = trainTargetReshaped2
                        	truthCopy = trainTargetCopy
                       	else:
                        	guessIn = validDataReshaped
                        	guessOut = validTargetReshaped
                        	truthCopy = validTargetCopy
                        guesses = sess.run(softmax_acc, feed_dict={X: guessIn, y: guessOut})
                        accuracy = 1 - (((np.absolute(guesses - truthCopy)).clip(0, 1).sum())/guesses.shape[0])
                       	entropy_loss = sess.run(temp5, feed_dict={X: guessIn, y: guessOut})/guesses.shape[0]
                       	cross_loss.append(entropy_loss)
                        loss_array.append(accuracy*100)
                       	print("accuracy: ", accuracy)
                       	'''
                       	for typeAcc, guessIn, guessOut, truthCopy, accuracy_array, loss_array in \
                       		zip(["train", "valid", "test"],
                                [trainDataReshaped2, validDataReshaped, testDataReshaped], 
                       			[trainTargetReshaped2, validTargetReshaped, testTargetReshaped], 
                       			[trainTargetCopy, validTargetCopy, testTargetCopy], 
                       			[train_accuracy, valid_accuracy, test_accuracy], 
                       			[train_loss, valid_loss, test_loss]):
                       		guesses = sess.run(softmax_acc, feed_dict={X: guessIn, y: guessOut})
	                        accuracy = 1 - (((np.absolute(guesses - truthCopy)).clip(0, 1).sum())/guesses.shape[0])
	                       	entropy_loss = sess.run(temp5, feed_dict={X: guessIn, y: guessOut})/guesses.shape[0]
	                       	loss_array.append(entropy_loss)
	                        accuracy_array.append(accuracy*100)
	                       	print("Epoch: ", epoch, " type: ", typeAcc, " with accuracy: ", accuracy)

                    sess.run(optim, feed_dict={X: miniBatchData, y: miniBatchTarget})
        '''
        trainplt = ax1.plot(epoch_array, train_accuracy, 'go', label="Train Accuracy")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel("Accuracy %", color='b')
        validplt = ax1.plot(epoch_array, valid_accuracy, 'ro', label="Valid Accuracy")
        testplt = ax1.plot(epoch_array, test_accuracy, 'bo', label="Test Accuracy")
        #plt.legend([trainplt, validplt, testplt], ["Train Acc", "Valid Acc", "Test Acc"])
        ax1.legend(loc="upper right")
        #ax2 = ax1.twinx()
        #ax1.plot(epoch_array, cross_loss, 'r.')
        #ax2.set_ylabel("Loss", color='r')
        fig.tight_layout()
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss ", color='r')
        trainloss = ax2.plot(epoch_array, train_loss, 'go', label="Train Loss")
        validloss = ax2.plot(epoch_array, valid_loss, 'ro', label="Valid Loss")
        testloss = ax2.plot(epoch_array, test_loss, 'bo', lable="Test Loss")
        ax2.legend(loc="upper right")
        plt.show()
        '''
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel("Accuracy %", color='b')
        trainplt = ax1.plot(epoch_array, train_accuracy, 'go', label="Train Accuracy")
        validplt = ax1.plot(epoch_array, valid_accuracy, 'ro', label="Valid Accuracy")
        testplt = ax1.plot(epoch_array, test_accuracy, 'bo', label="Test Accuracy")
        ax1.legend(loc="upper right")

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel("Loss", color='b')
        trainplt = ax2.plot(epoch_array, train_loss, 'go', label="Train Loss")
        validplt = ax2.plot(epoch_array, valid_loss, 'ro', label="Valid Loss")
        testplt = ax2.plot(epoch_array, test_loss, 'bo', label="Test Loss")
        ax2.legend(loc="upper right")

        plt.show()

        