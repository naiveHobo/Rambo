import os
import sys
import numpy as np
from layers import *


class Rambo:

    def __init__(self, config, data=None):
        self.mode = config.mode
        self.model = config.model
        self.train_dir = config.train_dir
        self.data_dir = config.data_dir
        self.log_dir = config.log_dir
        self.batch_size = np.int(config.batch_size)
        self.epochs = config.epochs
        self.dropout = config.dropout
        self.beta = config.beta_l2
        self.learning_rate = config.learning_rate
        self.resume = config.resume
        self.visualize = config.visualize
        self.best_val_loss = None
        self.save_model = config.save_model
        self.inp_dict = {}

        assert os.path.isfile(os.path.join(self.data_dir, 'X_train_mean.npy')), "X_train_mean.npy does not exist"
        self.x_mean = np.load(os.path.join(self.data_dir, 'X_train_mean.npy'))

        if self.mode == 'train':
            self.X_train, self.y_train, self.X_test, self.y_test = data
            self.num_batch = int(self.X_train.shape[0]) // self.batch_size
            self.build_train_graph()

        self.current_epoch = 0
        self.current_step = 0
        if self.resume or self.mode == 'test':
            if os.path.isfile(os.path.join(self.train_dir, 'save.npy')):
                self.current_epoch, self.current_step = np.load(os.path.join(self.train_dir, 'save.npy'))
            else:
                print("\nNo checkpoints, restarting training.\n")
                self.resume = 0

        self.build_model()

        if self.mode == 'test':
            self.resume = True
            self.sess = self.init_graph()

    def _normalize_func(self, image, label):
        tmp = tf.cast(image, dtype=tf.float32)
        tmp -= self.x_mean
        tmp /= 255.0
        tmp = tf.squeeze(tmp)
        return tmp, label

    def build_train_graph(self):
        with tf.variable_scope('input_pipeline'):
            self.inp_dict = {
                'X_data': tf.placeholder(dtype=tf.float32, shape=[None, self.X_train.shape[1], self.X_train.shape[2],
                                                                  self.X_train.shape[3]]),
                'y_data': tf.placeholder(dtype=tf.float32, shape=[None, self.y_train.shape[-1]])
            }

            train_data = tf.data.Dataset.from_tensor_slices((self.inp_dict['X_data'], self.inp_dict['y_data']))
            train_data = train_data.map(self._normalize_func)
            train_data = train_data.repeat()
            train_data = train_data.shuffle(buffer_size=32)
            train_data = train_data.batch(self.batch_size)

            val_data = tf.data.Dataset.from_tensor_slices((self.inp_dict['X_data'], self.inp_dict['y_data']))
            val_data = val_data.map(self._normalize_func)
            val_data = val_data.batch(self.batch_size)

            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

            self.train_init_op = iterator.make_initializer(train_data)
            self.val_init_op = iterator.make_initializer(val_data)

            self.features, self.labels = iterator.get_next()

    def build_model(self):

        with tf.variable_scope("Rambo"):
            if self.mode == 'test':
                self.features = input_layer("input_layer", [None, 192, 256, 4])

            if self.model == 'nvidia1':
                self.output = self.nvidia1_graph(solo=True)
            elif self.model == 'nvidia2':
                self.output = self.nvidia2_graph(solo=True)
            elif self.model == 'comma':
                self.output = self.comma_graph(solo=True)
            else:
                f1 = self.nvidia1_graph(solo=False)
                f2 = self.nvidia2_graph(solo=False)
                f3 = self.comma_graph(solo=False)
                merged = tf.concat([f1, f2, f3], axis=-1, name="merged")
                self.output = fully_connected("output", merged, num_output=1)

    def nvidia1_graph(self, solo=False):

        with tf.variable_scope("AutoPilot1"):
            conv1 = conv2d("conv1", self.features, filters=24, filter_size=(5, 5), stride=(2, 2), padding='SAME')
            relu1 = activation("relu1", conv1, function='relu')
            conv2 = conv2d("conv2", relu1, filters=36, filter_size=(5, 5), stride=(2, 2), padding='SAME')
            relu2 = activation("relu2", conv2, function='relu')
            conv3 = conv2d("conv3", relu2, filters=48, filter_size=(5, 5), stride=(2, 2), padding='SAME')
            relu3 = activation("relu3", conv3, function='relu')
            conv4 = conv2d("conv4", relu3, filters=64, filter_size=(3, 3), stride=(2, 2), padding='SAME')
            relu4 = activation("relu4", conv4, function='relu')
            conv5 = conv2d("conv5", relu4, filters=64, filter_size=(3, 3), stride=(2, 2), padding='SAME')
            relu5 = activation("relu5", conv5, function='relu')
            conv6 = conv2d("conv6", relu5, filters=64, filter_size=(3, 3), stride=(2, 2), padding='SAME')
            relu6 = activation("relu6", conv6, function='relu')
            conv6_f = flatten("conv6_f", relu6)
            fc7 = fully_connected("fc7", conv6_f, num_output=100)
            relu7 = activation("relu7", fc7, function='relu')
            fc8 = fully_connected("fc8", relu7, num_output=50)
            relu8 = activation("relu8", fc8, function='relu')
            fc9 = fully_connected("fc9", relu8, num_output=10)
            relu9 = activation("relu9", fc9, function='relu')
            if solo:
                output = fully_connected("output", relu9, num_output=1)
            else:
                output = relu9

            if self.visualize:
                with tf.variable_scope('visualize'):
                    relu1_avg = tf.reduce_mean(relu1, axis=3, keep_dims=True)
                    relu2_avg = tf.reduce_mean(relu2, axis=3, keep_dims=True)
                    relu3_avg = tf.reduce_mean(relu3, axis=3, keep_dims=True)
                    relu4_avg = tf.reduce_mean(relu4, axis=3, keep_dims=True)
                    relu5_avg = tf.reduce_mean(relu5, axis=3, keep_dims=True)
                    relu6_avg = tf.reduce_mean(relu6, axis=3, keep_dims=True)

                    deconv6 = deconv2d("deconv6", relu6_avg,
                                       out_shape=[1, relu5.get_shape()[1].value, relu5.get_shape()[2].value, 1],
                                       filter_size=(3, 3), stride=(2, 2), padding='SAME', kernel_init='constant',
                                       const_init=1.0)
                    conv6_to_5 = tf.multiply(deconv6, relu5_avg)
                    deconv5 = deconv2d("deconv5", conv6_to_5,
                                       out_shape=[1, relu4.get_shape()[1].value, relu4.get_shape()[2].value, 1],
                                       filter_size=(3, 3), stride=(2, 2), padding='SAME', kernel_init='constant',
                                       const_init=1.0)
                    conv5_to_4 = tf.multiply(deconv5, relu4_avg)
                    deconv4 = deconv2d("deconv4", conv5_to_4,
                                       out_shape=[1, relu3.get_shape()[1].value, relu3.get_shape()[2].value, 1],
                                       filter_size=(3, 3), stride=(2, 2), padding='SAME', kernel_init='constant',
                                       const_init=1.0)
                    conv4_to_3 = tf.multiply(deconv4, relu3_avg)
                    deconv3 = deconv2d("deconv3", conv4_to_3,
                                       out_shape=[1, relu2.get_shape()[1].value, relu2.get_shape()[2].value, 1],
                                       filter_size=(3, 3), stride=(2, 2), padding='SAME', kernel_init='constant',
                                       const_init=1.0)
                    conv3_to_2 = tf.multiply(deconv3, relu2_avg)
                    deconv2 = deconv2d("deconv2", conv3_to_2,
                                       out_shape=[1, relu1.get_shape()[1].value, relu1.get_shape()[2].value, 1],
                                       filter_size=(3, 3), stride=(2, 2), padding='SAME', kernel_init='constant',
                                       const_init=1.0)
                    conv2_to_1 = tf.multiply(deconv2, relu1_avg)
                    deconv1 = deconv2d("deconv1", conv2_to_1,
                                       out_shape=[1, self.features.get_shape()[1].value,
                                                  self.features.get_shape()[2].value, 1],
                                       filter_size=(3, 3), stride=(2, 2), padding='SAME', kernel_init='constant',
                                       const_init=1.0)

                    self.vis_mask = tf.squeeze(tf.div(
                        tf.subtract(deconv1, tf.reduce_min(deconv1)),
                        tf.subtract(tf.reduce_max(deconv1), tf.reduce_min(deconv1))),
                        axis=0, name="vis_mask")

            return output

    def nvidia2_graph(self, solo=False):

        with tf.variable_scope("AutoPilot2"):
            conv1 = conv2d("conv1", self.features, filters=24, filter_size=(5, 5), stride=(2, 2), padding='SAME')
            relu1 = activation("relu1", conv1, function='relu')
            conv2 = conv2d("conv2", relu1, filters=36, filter_size=(5, 5), stride=(2, 2), padding='SAME')
            relu2 = activation("relu2", conv2, function='relu')
            conv3 = conv2d("conv3", relu2, filters=48, filter_size=(5, 5), stride=(2, 2), padding='SAME')
            relu3 = activation("relu3", conv3, function='relu')
            conv4 = conv2d("conv4", relu3, filters=64, filter_size=(3, 3), stride=(2, 2), padding='SAME')
            relu4 = activation("relu4", conv4, function='relu')
            conv5 = conv2d("conv5", relu4, filters=64, filter_size=(3, 3), stride=(2, 2), padding='SAME')
            relu5 = activation("relu5", conv5, function='relu')
            conv5_f = flatten("conv5_f", relu5)
            fc6 = fully_connected("fc6", conv5_f, num_output=100)
            relu6 = activation("relu6", fc6, function='relu')
            fc7 = fully_connected("fc7", relu6, num_output=50)
            relu7 = activation("relu7", fc7, function='relu')
            fc8 = fully_connected("fc8", relu7, num_output=10)
            relu8 = activation("relu8", fc8, function='relu')
            if solo:
                output = fully_connected("output", relu8, num_output=1)
            else:
                output = relu8

            return output

    def comma_graph(self, solo=False):

        with tf.variable_scope("Comma"):
            conv1 = conv2d("conv1", self.features, filters=16, filter_size=(8, 8), stride=(4, 4), padding='SAME')
            relu1 = activation("relu1", conv1, function='relu')
            conv2 = conv2d("conv2", relu1, filters=32, filter_size=(5, 5), stride=(2, 2), padding='SAME')
            relu2 = activation("relu2", conv2, function='relu')
            conv3 = conv2d("conv3", relu2, filters=64, filter_size=(5, 5), stride=(2, 2), padding='SAME')
            relu3 = activation("relu3", conv3, function='relu')
            conv3_f = flatten("conv3_f", relu3)
            fc4 = fully_connected("fc4", conv3_f, num_output=512)
            relu4 = activation("relu4", fc4, function='relu')
            if solo:
                output = fully_connected("output", relu4, num_output=1)
            else:
                output = relu4

            return output

    def train(self):
        self.loss = tf.reduce_mean(tf.square(self.labels - self.output))
        
        global_step = tf.Variable(self.current_step, name='global_step', dtype=tf.int64)
        saver = tf.train.Saver(max_to_keep=10)

        with tf.variable_scope('Optimizer'):
            starter_learning_rate = self.learning_rate
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step, 1000, 0.95, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=global_step)

        train_summary = tf.summary.scalar('train_loss', self.loss)
        val_summary = tf.summary.scalar('val_loss', self.loss)

        with tf.Session() as sess:
            print("\nInitializing training")
            sess.run(tf.global_variables_initializer())

            if self.resume:
                print("\nLoading previously trained model")
                print("{} out of {}, epochs completed in previous run.".format(self.current_epoch, self.epochs))
                try:
                    ckpt_file = os.path.join(self.train_dir, "model.ckpt-" + str(self.current_step))
                    saver.restore(sess, ckpt_file)
                    print("\nResuming training")
                except Exception as e:
                    print(str(e).split('\n')[0])
                    print("\nCheckpoints not found")
                    sys.exit(0)

            writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

            for epoch in range(self.current_epoch, self.epochs):
                loss = []

                sess.run(self.train_init_op,
                         feed_dict={self.inp_dict['X_data']: self.X_train, self.inp_dict['y_data']: self.y_train})

                for batch_idx in range(self.num_batch):
                    run = [global_step, optimizer, self.loss, train_summary]
                    step, _, current_loss, train_summ = sess.run(run)
                    if step % 10 == 0:
                        writer.add_summary(train_summ, step)
                    print("{}: Global step: {}\tLoss: {}".format(epoch, step, current_loss))
                    loss.append(current_loss)

                sess.run(self.val_init_op,
                         feed_dict={self.inp_dict['X_data']: self.X_test, self.inp_dict['y_data']: self.y_test})
                val_loss, val_summ = sess.run([self.loss, val_summary])
                writer.add_summary(val_summ, step)

                print("\nAverage training Loss: {},\tValidation loss: {}\n".format(np.mean(loss), val_loss))

                if self.best_val_loss is None:
                    self.best_val_loss = val_loss
                elif val_loss < self.best_val_loss or epoch == self.epochs - 1:
                    self.best_val_loss = val_loss
                    print("Saving Model..\n")
                    saver.save(sess, os.path.join(self.train_dir, "model.ckpt"), global_step=global_step)
                    np.save(os.path.join(self.train_dir, "save"), (epoch, step))

        print("Training complete")

    def init_graph(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        if self.resume:
            try:
                ckpt_file = os.path.join(self.train_dir, "model.ckpt-" + str(self.current_step))
                saver.restore(sess, ckpt_file)
            except Exception as e:
                print(str(e).split('\n')[0])
                print("\nCheckpoints not found")
                sys.exit(0)
        if self.save_model:
            saver.save(sess, "./saved_model/model.ckpt")
            print("\nModel saved as saved_model/model.ckpt")
        return sess

    def predict(self, image, visualize=False):
        image = image.astype('float32')
        image -= self.x_mean
        image /= 255.0
        if visualize:
            output, mask = self.sess.run([self.output, self.vis_mask], feed_dict={self.features: image})
            return output, mask
        else:
            output = self.sess.run(self.output, feed_dict={self.features: image})
            return output
