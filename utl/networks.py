import glob
import os
import re
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input,Flatten, Dense, Dropout,Add,Average,LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from utl.custom_layers import NeighborAggregator, Graph_Attention, Last_Sigmoid, NeighborAttention,TransformerBlock,DistanceLayer,multiply, Score_pooling, Feature_pooling, RC_block, DP_pooling
from tensorflow.keras.regularizers import l2
from args import parse_args
from utl.DataGenerator import DataGenerator
from utl.siamese_pairs import get_siamese_pairs,SiameseGenerator, parallel_get_siamese_pairs
from utl.custom_layers import NeighborAggregator, Graph_Attention, Last_Sigmoid, DistanceLayer,multiply, Score_pooling, Feature_pooling, RC_block, DP_pooling
from utl.dataset import Get_train_valid_Path
from utl.metrics import bag_accuracy, bag_loss
from utl.metrics import get_contrastive_loss, siamese_accuracy
from utl.stack_layers import stack_layers, make_layer_list
from utl.BreastCancerDataset import BreastCancerDataset
from utl.ColonCancerDataset import ColonCancerDataset
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.callbacks import CallbackList
import time
import tensorflow as tf


class SiameseNet:
    def __init__(self, args, useMulGpue=False):
        """
        Build the architecture of the siamese net
        Parameters
        ----------
        arch:          a dict describing the architecture of the neural network to be trained
        containing input_types and input_placeholders for each key and value pair, respecively.
        input_shape:   tuple(int, int, int) of the input shape of the patches
        useMulGpue:    boolean, whether to use multi-gpu processing or not
        """

        self.input_shape = tuple(args.input_shape)
        self.args = args
        self.arch = args.arch
        self.siam_k = args.siam_k
        self.experiment_name = args.experiment_name
        self.weight_decay = args.weight_decay
        self.pooling_mode = args.pooling_mode
        self.init_lr = args.init_lr
        self.epochs = args.epochs
        self.useGated=args.useGated
        self.siamese_weights_path=args.siamese_weights_path
        self.siam_pixel_dist=args.siam_pixel_dist
        self.siam_epochs=args.siam_epochs
        self.siam_batch_size=args.siam_batch_size

        self.inputs = {
            'left_input': Input(self.input_shape),
            'right_input': Input(self.input_shape),
        }

        self.useMulGpu = useMulGpue
        self.layers = []
        self.layers += make_layer_list(self.arch, 'siamese', args.weight_decay)

        self.outputs = stack_layers(self.inputs, self.layers)

        self.distance = DistanceLayer(output_dim=1)([self.outputs["left_input"], self.outputs["right_input"]])

        self.net = Model(inputs=[self.inputs["left_input"], self.inputs["right_input"]], outputs=[self.distance])

        self.net.compile(optimizer=Adam(lr=args.siam_init_lr, beta_1=0.9, beta_2=0.999),
                             loss=get_contrastive_loss(m_neg=1, m_pos=0.05), metrics=[siamese_accuracy])


    def train(self, train_bags,val_bags, irun, ifold):
        """
        Train the siamese net

        Parameters
        ----------
        pairs_train : a list of lists, each of which contains an np.ndarray of the patches of each image,
        the label of each image and a list of filenames of the patches
        check_dir   : str, specifying the directory where weights of the siamese net are going to be stored
        irun        : int reffering to the id of the experiment
        ifold       : fold reffering to the fold of the k-cross fold validation

        Returns
        -------
        A History object containing a record of training loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values

        """

        bags=np.concatenate((train_bags, val_bags))
        train_pairs, train_labels = get_siamese_pairs(bags, k=self.siam_k, pixel_distance=self.siam_pixel_dist,augmentation=True)
        train_gen = SiameseGenerator(train_pairs, train_labels, batch_size=self.siam_batch_size, dim=self.input_shape,
                                     shuffle=True)
        # val_pairs, val_labels = get_siamese_pairs(val_bags, k=self.siam_k, pixel_distance=self.siam_pixel_dist,
        #                                           augmentation=False)
        # val_gen = SiameseGenerator(val_pairs, val_labels, batch_size=self.siam_batch_size, dim=self.input_shape,
        #                            shuffle=False)

        if not os.path.exists(self.siamese_weights_path):
            os.makedirs(self.siamese_weights_path)

        filepath = os.path.join(self.siamese_weights_path,
                                "weights-irun:{}-ifold:{}".format(irun, ifold) + ".hdf5")

        checkpoint_fixed_name = ModelCheckpoint(filepath,
                                                monitor='loss', verbose=1, save_best_only=True,
                                                save_weights_only=False, mode='auto', save_freq='epoch')

        EarlyStop = EarlyStopping(monitor='loss', patience=20)

        callbacks = [checkpoint_fixed_name, EarlyStop]


        self.net.fit_generator(generator=train_gen, steps_per_epoch=len(train_gen) ,
                               epochs=self.siam_epochs,
                                callbacks=callbacks
                               )

        return self.net


class GraphAttnet:
    def __init__(self, args, useMulGpue=False):
        """
        Build the architercure of the Graph Att net
        Parameters
        ----------
        arch:          a dict describing the architecture of the neural network to be trained
        mode            :str, specifying the version of the model (siamese, euclidean)
        containing input_types and input_placeholders for each key and value pair, respecively.
        input_shape:   tuple(int, int, int) of the input shape of the patches
        useMulGpue:    boolean, whether to use multi-gpu processing or not
        """


        self.mode=args.mode
        self.input_shape = tuple(args.input_shape)
        self.prob = args.prob
        self.args = args
        self.arch = args.arch
        self.mode = args.mode
        self.input_shape = tuple(args.input_shape)
        self.data = args.data
        self.weight_file = args.weight_file
        self.k = args.k
        self.save_dir = args.save_dir
        self.experiment_name = args.experiment_name
        self.weight_decay = args.weight_decay
        self.pooling_mode = args.pooling_mode
        self.init_lr = args.init_lr
        self.epochs = args.epochs
        self.useGated=args.useGated
        self.siamese_weights_path=args.siamese_weights_path


        self.inputs = {
            'bag': Input(self.input_shape),
            'adjacency_matrix': Input(shape=(None,), dtype='float32', name='adjacency_matrix'),
        }

        self.useMulGpu = useMulGpue
        self.layers = []
        self.layers += make_layer_list(self.arch, 'graph', self.weight_decay)

        self.outputs = stack_layers(self.inputs, self.layers)

        # neigh = Graph_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(self.weight_decay),
        #                       name='neigh',
        #                       use_gated=args.useGated)(self.outputs["bag"])
        # neigh = MultiHeadAttention(d_model=128, num_heads=1)(self.outputs["bag"])

        # alpha = NeighborAggregator(output_dim=1, name="alpha")([neigh, self.inputs["adjacency_matrix"]])
        #
        # attention_output = multiply([alpha, self.outputs["bag"]], name="mul")
        attention_output, attention_weights = NeighborAttention(embed_dim=256)(
            [self.outputs["bag"], self.inputs["adjacency_matrix"]])

        # attention_output=TransformerBlock(embed_dim=256, ff_dim=256, training=self.training)([self.outputs["bag"], self.inputs["adjacency_matrix"]])

        out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid', pooling_mode=self.pooling_mode)(attention_output)

        self.net = Model(inputs=[self.inputs["bag"], self.inputs["adjacency_matrix"]], outputs=[out])

        self.net.compile(optimizer=Adam(lr=self.init_lr, beta_1=0.9, beta_2=0.999), loss=bag_loss,
                         metrics=[bag_accuracy])
    @property
    def model(self):
        return self.net

    def load_siamese(self, irun, ifold):
        """
        Loads the appropriate siamese model using the information of the fold of k-cross
        fold validation and the id of experiment
        Parameters
        ----------
        check_dir  : directory where the weights of the pretrained siamese network. Weight files are stored in the format:
        weights-irun:d-ifold:d.hdf5
        irun       : int referring to the id of the experiment
        ifold      : int referring to the fold from the k-cross fold validation

        Returns
        -------
        returns  a Keras model instance of the pre-trained siamese net
        """

        def extract_number(f):
            s = re.findall("\d+\.\d+", f)
            return ((s[0]) if s else -1, f)

        file_paths = glob.glob(os.path.join(self.siamese_weights_path, "weights-irun:{}-ifold:{}*.hdf5".format(irun, ifold)))
        file_paths.reverse()
        file_path = (min(file_paths, key=extract_number))

        self.siamese_net = load_model(file_path, custom_objects={'DistanceLayer': DistanceLayer,
                                                                 "contrastive_loss": get_contrastive_loss(),
                                                                 "siamese_accuracy": siamese_accuracy})
        return self.siamese_net

    def train(self,train_bags ,irun, ifold,detection_model):
        """
        Train the Graph Att net
        Parameters
        ----------
        train_set       : a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches
        check_dir       :str, specifying directory where the weights of the siamese net are stored
        irun            :int, id of the experiment
        ifold           :int, fold of the k-corss fold validation
        weight_file     :boolen, specifying whether there is a weightflie or not

        Returns
        -------
        A History object containing  a record of training loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values.
        """


        train_bags, val_bags = Get_train_valid_Path(train_bags, ifold, train_percentage=0.9)
        if self.data=='colon':
            model_val_set = ColonCancerDataset(patch_size=27,augmentation=False).load_bags(val_bags)
            model_train_set = ColonCancerDataset(patch_size=27,augmentation=True).parallel_load_bags(train_bags)
        else :

            model_val_set=BreastCancerDataset(format='.tif', patch_size=128,
                                stride=16, augmentation=False, model=detection_model).load_bags(wsi_paths=val_bags)
            model_train_set = BreastCancerDataset(format='.tif', patch_size=128,
                                               stride=16, augmentation=True, model=detection_model).load_bags(wsi_paths=train_bags)

        if self.mode=="siamese":
            if not self.weight_file:
                self.siamese_net = SiameseNet(self.args, useMulGpue=False)
                self.siamese_net.train(model_train_set, model_val_set, irun=irun,
                                       ifold=ifold)


            self.trained_model = self.load_siamese(irun, ifold)


            train_gen = DataGenerator(prob=self.prob,batch_size=1, data_set=model_train_set, k=self.k, shuffle=True, mode=self.mode,
                                      trained_model=self.trained_model)

            val_gen = DataGenerator(prob=self.prob,batch_size=1, data_set=model_val_set, k=self.k, shuffle=False, mode=self.mode,
                                    trained_model=self.trained_model)
        else:
            train_gen = DataGenerator(prob=self.prob,batch_size=1, data_set=model_train_set, k=self.k, shuffle=True, mode=self.mode)

            val_gen = DataGenerator(prob=self.prob,batch_size=1, data_set=model_val_set, k=self.k, shuffle=False, mode=self.mode)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        checkpoint_path = os.path.join(self.save_dir, self.experiment_name + ".hdf5")


        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                         monitor='val_loss',
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         mode='auto',
                                                         save_freq='epoch',
                                                         verbose=1)

        _callbacks = [EarlyStopping(monitor='val_loss', patience=20), cp_callback]
        callbacks = CallbackList(_callbacks, add_history=True, model=self.net)

        logs = {}
        callbacks.on_train_begin(logs=logs)

        optimizer = Adam(learning_rate=self.init_lr, beta_1=0.9, beta_2=0.999)
        loss_fn = BinaryCrossentropy(from_logits=False)
        train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        val_acc_metric = tf.keras.metrics.BinaryAccuracy()

        @tf.function(experimental_relax_shapes=True)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = self.net(x, training=True)
                loss_value = loss_fn(y, logits)
            grads = tape.gradient(loss_value, self.net.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
            train_loss_tracker.update_state(loss_value)
            train_acc_metric.update_state(y, logits)
            return {"train_loss": train_loss_tracker.result(), "train_accuracy": train_acc_metric.result()}

        @tf.function(experimental_relax_shapes=True)
        def val_step(x, y):
            val_logits = self.net(x, training=False)
            val_loss = loss_fn(y, val_logits)
            val_loss_tracker.update_state(val_loss)
            val_acc_metric.update_state(y, val_logits)
            return {"val_loss": val_loss_tracker.result(), "val_accuracy": val_acc_metric.result()}

        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(train_gen):

                callbacks.on_batch_begin(step, logs=logs)
                callbacks.on_train_batch_begin(step, logs=logs)
                train_dict = train_step(x_batch_train, np.expand_dims(y_batch_train, axis=0))

                logs["train_loss"] = train_dict["train_loss"]

                callbacks.on_train_batch_end(step, logs=logs)
                callbacks.on_batch_end(step, logs=logs)
                if step % 20 == 0:
                    print("Training loss at step %d: %.4f" % (step, train_dict["train_loss"]))

            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            train_acc_metric.reset_states()
            train_loss_tracker.reset_states()

            for step, (x_batch_val, y_batch_val) in enumerate(val_gen):
                callbacks.on_batch_begin(step, logs=logs)
                callbacks.on_test_batch_begin(step, logs=logs)
                val_dict = val_step(x_batch_val, np.expand_dims(y_batch_val, axis=0))
                logs["val_loss"] = val_dict["val_loss"]

                callbacks.on_test_batch_end(step, logs=logs)
                callbacks.on_batch_end(step, logs=logs)

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            val_loss_tracker.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))
            callbacks.on_epoch_end(epoch, logs=logs)

        callbacks.on_train_end(logs=logs)

    def predict(self, test_bags, detection_model, test_model, irun, ifold):

        """
        Evaluate the test set
        Parameters
        ----------
        test_set: a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches

        Returns
        -------

        test_loss : float reffering to the test loss
        acc       : float reffering to the test accuracy
        precision : float reffering to the test precision
        recall    : float referring to the test recall
        auc       : float reffering to the test auc


        """

        if self.data == "colon":
            test_set = ColonCancerDataset(patch_size=27, augmentation=False).load_bags(wsi_paths=test_bags)
        else:
            test_set = BreastCancerDataset(format='.tif', patch_size=128,
                                           stride=16, augmentation=False, model=detection_model).load_bags(
                wsi_paths=test_bags)

        if self.mode == "siamese":
            self.discriminator_test = self.load_siamese(irun, ifold)
            test_gen = DataGenerator(prob=self.prob, batch_size=1, data_set=test_set, k=self.k, shuffle=False,
                                     mode=self.mode,
                                     trained_model=self.discriminator_test)
        else:
            test_gen = DataGenerator(prob=self.prob, batch_size=1, data_set=test_set, k=self.k, shuffle=False,
                                     mode=self.mode)

        loss_value = []
        test_loss_fn = BinaryCrossentropy(from_logits=False)
        eval_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        checkpoint_path = os.path.join(self.save_dir, self.experiment_name + ".hdf5")
        test_model.load_weights(checkpoint_path)

        @tf.function(experimental_relax_shapes=True)
        def test_step(images, labels):

            predictions = test_model(images, training=False)
            test_loss = test_loss_fn(labels, predictions)

            eval_accuracy_metric.update_state(labels, predictions)
            return test_loss, predictions

        y_pred = []
        y_true = []
        for x_batch_val, y_batch_val in test_gen:
            test_loss, pred = test_step(x_batch_val, np.expand_dims(y_batch_val, axis=0))
            loss_value.append(test_loss)
            y_true.append(y_batch_val)
            y_pred.append(pred.numpy().tolist()[0][0])

        test_loss = np.mean(loss_value)
        print("Test loss: %.4f" % (float(test_loss),))

        test_acc = eval_accuracy_metric.result()
        print("Test acc: %.4f" % (float(test_acc),))

        auc = roc_auc_score(y_true, y_pred)
        print("AUC {}".format(auc))

        precision = precision_score(y_true, np.round(np.clip(y_pred, 0, 1)))
        print("precision {}".format(precision))

        recall = recall_score(y_true, np.round(np.clip(y_pred, 0, 1)))
        print("recall {}".format(recall))

        return test_loss, test_acc, auc, precision, recall


    # def visualize_conv_layer(self,layer_name, data_name, test_img, detection_model, irun, ifold,saved_weights_dir=None,check_dir=None):
    #
    #
    #     if data_name == "colon":
    #         test_set = ColonCancerDataset(patch_size=27, augmentation=False).load_bags(wsi_paths=[test_img])
    #     else:
    #         test_set = BreastCancerDataset(format='.tif', patch_size=128,
    #                                        stride=16, augmentation=False, model=detection_model).load_bags(wsi_paths=test_img)
    #
    #     if self.mode == "siamese":
    #
    #         self.siamese_net = self.load_siamese(check_dir, irun, ifold)
    #         test_gen = DataGenerator(batch_size=1, data_set=test_set, k=args.k, shuffle=False, mode=self.mode,
    #                                  siamese_model=self.siamese_net)
    #     else:
    #         test_gen = DataGenerator(batch_size=1, data_set=test_set, k=args.k, shuffle=False, mode=self.mode)
    #
    #     layer_output = self.net.get_layer(layer_name).output
    #
    #     intermediate_model = Model(inputs=self.net.input, outputs=layer_output)
    #     intermediate_model.load_weights(saved_weights_dir, by_name=True)
    #
    #     intermediate_prediction = intermediate_model.predict_on_batch(test_gen[0][0])
    #
    #     return intermediate_prediction