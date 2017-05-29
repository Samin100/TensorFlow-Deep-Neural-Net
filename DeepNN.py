'''
MNIST Digit Recognizer using a deep neural net with 3 hidden layers.

Basic flow of data:
input -> weight -> hidden layer 1 (activation function) -> weights -> hidden layer 2 (activation function) -> 
weights -> output layer

Then we compare the output to the intended output (cross entropy).
Then we use an optimization function (optimizer) to minimize cost (AdamOptimizer... stochastic gradient descent, AdaGrad).
This backwards manipulation of weights is backpropogation.

One cycle of passing the data through the net (feeding forward) then adjusting weights (backpropagation) = 1 epoch.
'''


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

'''
10 classes, one for each digit 0-9

Example of one_hot:
0 = [1,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0]...
'''

# hyperparameters
hl1_num_nodes = 500
hl2_num_nodes = 500
hl3_num_nodes = 500

num_classes = 10

# process batches of 100 images at a time, useful if dataset can't fit into memory
batch_size = 100


'''Input data placeholder, image is 28 x 28 pixels wide, but we will flatten it to be a 1 dimensional array of 
784 (height x width). Then we're explicitly defining what the input should be. If the input is not what we declare 
it then TF will throw an error, explicit is better than implicit.
'''
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):

    # Initializing our tensors with random weights and biases

    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal(shape=[784, hl1_num_nodes])),
                      'biases': tf.Variable(tf.random_normal([hl1_num_nodes]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal(shape=[hl1_num_nodes, hl2_num_nodes])),
                      'biases': tf.Variable(tf.random_normal([hl2_num_nodes]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal(shape=[hl2_num_nodes, hl3_num_nodes])),
                      'biases': tf.Variable(tf.random_normal([hl3_num_nodes]))}

    output_layer = {'weights': tf.Variable(tf.random_normal(shape=[hl3_num_nodes, num_classes])),
                    'biases': tf.Variable(tf.random_normal([num_classes]))}


    # Format is (input_data * weights) + biases
    # Then apply the ReLu activation function

    l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add((tf.matmul(l3, output_layer['weights'])), output_layer['biases'])

    return output


def train_neural_network(x):

    prediction = neural_network_model(x)

    # using cross entropy with logits as our cost function (since we're using one hot output)
    # essentially comparing our output to the intended output
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    # we want to minimize our cost
    # default learning rate for the optimizer is 0.001, but explicit is better than implicit
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    # how many epochs: cycles of feed forward + backprop (adjusting weights and biases)
    num_epochs = 10

    # start of our TF session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):

            epoch_loss = 0

            # total number of samples / batch size
            for _ in range(int(mnist.train.num_examples/batch_size)):

                # optimizing the weights and biases. TF does this for us.
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # checking our cost
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', num_epochs, 'Loss:', epoch_loss)

        # seeing if our prediction is equal to y
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
