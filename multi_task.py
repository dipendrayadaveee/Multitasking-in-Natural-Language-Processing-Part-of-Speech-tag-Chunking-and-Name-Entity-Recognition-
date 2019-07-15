from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import argparse
import saveload
import collections
import os
import pandas as pd
import csv
import pickle
import time


########################################################################################################
########################################################################################################
#####################Model Reader#######################################################################
########################################################################################################


"""Utilities for parsing CONll text files."""

"""
    1.0. Utility Methods
"""


def read_tokens(filename, padding_val, col_val=-1):
    # Col Values
    # 0 - words
    # 1 - POS
    # 2 - tags
    # 3 - NER tags
#opens the filename in 'read and text mode'
    with open(filename, 'rt', encoding='utf8') as csvfile:
            #reads file from csvfile and the words are seperated by space
            r = csv.reader(csvfile, delimiter=' ')
            # mAx = 0
            # mIn= 100
            # for row in r:
            #     if (len(row))>mAx:
            #         mAx=(len(row))
            #     if (len(row))<mIn and len(row)>0:
            #         mIn=(len(row))
            # print(mAx)
            # print(mIn)
            words = np.transpose(np.asarray([x for x in r if (x != [] and len(x)==4) ]  )).astype(object)

    print(words)
    # padding token '0'
    print('reading ' + str(col_val) + ' ' + filename)
    if col_val!=-1:
        words = words[col_val]
        #np.pad, pads the words with padwidth(front padding_val, end no pad)
    print(words)
    return np.pad(
        words, pad_width=(padding_val, 0), mode='constant', constant_values='0')

#Basically takes in the text file, with the req. padding width and col value, o is words, 1 is POS and 2 is Tags
def _build_vocab(filename, padding_width, col_val):
    # can be used for input vocab
    data = read_tokens(filename, padding_width, col_val)
    counter = collections.Counter(data)
    # get rid of all words with frequency == 1
    counter = {k: v for k, v in counter.items() if v > 1}
    counter['<unk>'] = 10000
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
# returns dictionary with word ids and
    return word_to_id

def _build_tags(filename, padding_width, col_val):
    # can be used for classifications and input vocab
    data = read_tokens(filename, padding_width, col_val)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1]) #makes all the data in counter in a list of pairs
    words, _ = list(zip(*count_pairs))  #takes the first number from the tuples count pairs
    tag_to_id = dict(zip(words, range(len(words))))#makes a dictionary of the words and its ids
    if col_val == 1:
        pickle.dump(tag_to_id,open('pos_to_id.pkl','wb')) #serializes the tag_to_id variable and stores it into pos_to_id.pkl file
        pickle.dump(count_pairs,open('pos_counts.pkl','wb'))

    return tag_to_id


"""
    1.1. Word Methods
"""


def _file_to_word_ids(filename, word_to_id, padding_width):
    # assumes _build_vocab has been called first as is called word to id
    data = read_tokens(filename, padding_width, 0)
    default_value = word_to_id['<unk>']
    return [word_to_id.get(word, default_value) for word in data]#returns ids for the word comparing it with the vocabulary

"""
    1.2. tag Methods
"""


def _int_to_tag(tag_int, tag_vocab_size):
    # creates the one-hot vector according to the vocabsize
    a = np.empty(tag_vocab_size)
    a.fill(0)
    np.put(a, tag_int, 1)
    return a


def _seq_tag(tag_integers, tag_vocab_size):
    # create the array of one-hot vectors for your sequence
    #so basically it contains all the one hot encoded vectors in a vertical stack
    return np.vstack(_int_to_tag(
                     tag, tag_vocab_size) for tag in tag_integers)


def _file_to_tag_classifications(filename, tag_to_id, padding_width, col_val):
    # assumes _build_vocab has been called first and is called tag to id
    data = read_tokens(filename, padding_width, col_val)
    return [tag_to_id[tag] for tag in data]


def raw_x_y_data(data_path, num_steps):
    train = "train.txt"
    valid = "validation.txt"
    train_valid = "train_val_combined.txt"
    comb = "all_combined.txt"
    test = "test.txt"

    train_path = os.path.join(data_path, train)
    valid_path = os.path.join(data_path, valid)
    train_valid_path = os.path.join(data_path, train_valid)
    comb_path = os.path.join(data_path, comb)
    test_path = os.path.join(data_path, test)

    # checking for all combined
    if not os.path.exists(data_path + '/train_val_combined.txt'):
        print('writing train validation combined')
        train_data = pd.read_csv(data_path + '/train.txt', sep= ' ',header=None)
        validation_data = pd.read_csv(data_path + '/validation.txt', sep= ' ',header=None)

        comb = pd.concat([train_data,validation_data])
        comb.to_csv(data_path + '/train_val_combined.txt', sep=' ', index=False, header=False)

    if not os.path.exists(data_path + '/all_combined.txt'):
        print('writing combined')
        test_data = pd.read_csv(data_path + '/test.txt', sep= ' ',header=None)
        train_data = pd.read_csv(data_path + '/train.txt', sep= ' ',header=None)
        val_data = pd.read_csv(data_path + '/validation.txt', sep=' ', header=None)

        comb = pd.concat([train_data,val_data,test_data])
        comb.to_csv(data_path + '/all_combined.txt', sep=' ', index=False, header=False)

    word_to_id = _build_vocab(train_path, num_steps-1, 0)
    # use the full training set for building the target tags
    pos_to_id = _build_tags(comb_path, num_steps-1, 1)

    chunk_to_id = _build_tags(comb_path, num_steps-1, 2)
    #all the datas from
    #ner_to_id =  _build_tags(comb_path, num_steps-1, 3)

    word_data_t = _file_to_word_ids(train_path, word_to_id, num_steps-1)
    pos_data_t = _file_to_tag_classifications(train_path, pos_to_id, num_steps-1, 1)
    chunk_data_t = _file_to_tag_classifications(train_path, chunk_to_id, num_steps-1, 2)

    word_data_v = _file_to_word_ids(valid_path, word_to_id, num_steps-1)
    pos_data_v = _file_to_tag_classifications(valid_path, pos_to_id, num_steps-1, 1)
    chunk_data_v = _file_to_tag_classifications(valid_path, chunk_to_id, num_steps-1, 2)

    word_data_c = _file_to_word_ids(train_valid_path, word_to_id, num_steps-1)
    pos_data_c = _file_to_tag_classifications(train_valid_path, pos_to_id, num_steps-1, 1)
    chunk_data_c = _file_to_tag_classifications(train_valid_path, chunk_to_id, num_steps-1, 2)

    word_data_test = _file_to_word_ids(test_path, word_to_id, num_steps-1)
    pos_data_test = _file_to_tag_classifications(test_path, pos_to_id, num_steps-1, 1)
    chunk_data_test = _file_to_tag_classifications(test_path, chunk_to_id, num_steps-1, 2)

    return word_data_t, pos_data_t, chunk_data_t, word_data_v, \
        pos_data_v, chunk_data_v, word_to_id, pos_to_id, chunk_to_id, \
        word_data_test, pos_data_test, chunk_data_test, word_data_c, \
        pos_data_c, chunk_data_c


def create_batches(raw_words, raw_pos, raw_chunk, batch_size, num_steps, pos_vocab_size,
                   chunk_vocab_size):
    """Tokenize and create batches From words (inputs), raw_pos (output 1), raw_chunk(output 2). The parameters
    of the minibatch are defined by the batch_size, the length of the sequence.

    :param raw_words:
    :param raw_pos:
    :param raw_chunk:
    :param batch_size:
    :param num_steps:
    :param pos_vocab_size:
    :param chunk_vocab_size:
    :return:
    """

    def _reshape_and_pad(tokens, batch_size, num_steps):
        tokens = np.array(tokens, dtype=np.int32)
        data_len = len(tokens)
        post_padding_required = (batch_size*num_steps) - np.mod(data_len, batch_size*num_steps)

        tokens = np.pad(tokens, (0, post_padding_required), 'constant',
                        constant_values=0)
        epoch_length = len(tokens) // (batch_size*num_steps)
        tokens = tokens.reshape([batch_size, num_steps*epoch_length])
        return tokens

    """
    1. Prepare the input (word) data
    """
    word_data = _reshape_and_pad(raw_words, batch_size, num_steps)
    pos_data = _reshape_and_pad(raw_pos, batch_size, num_steps)
    chunk_data = _reshape_and_pad(raw_chunk, batch_size, num_steps)

    """
    3. Do the epoch thing and iterate
    """
    data_len = len(raw_words)
    # how many times do you iterate to reach the end of the epoch
    epoch_size = (data_len // (batch_size*num_steps)) + 1

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = word_data[:, i*num_steps:(i+1)*num_steps]
        y_pos = np.vstack(_seq_tag(pos_data[tag, i*num_steps:(i+1)*num_steps],
                          pos_vocab_size) for tag in range(batch_size))
        y_chunk = np.vstack(_seq_tag(chunk_data[tag, i*num_steps:(i+1)*num_steps],
                            chunk_vocab_size) for tag in range(batch_size))
        y_pos = y_pos.astype(np.int32)
        y_chunk = y_chunk.astype(np.int32)
        yield (x, y_pos, y_chunk)


def _int_to_string(int_pred, d):

    # integers are the Values
    keys = []
    for x in int_pred:
        keys.append([k for k, v in d.items() if v == (x)])

    return keys


def res_to_list(res, batch_size, num_steps, to_id, w_length):

    tmp = np.concatenate([x.reshape(batch_size, num_steps)
                          for x in res], axis=1).reshape(-1)
    tmp = np.squeeze(_int_to_string(tmp, to_id))
    return tmp[range(num_steps-1, w_length)].reshape(-1,1)




#########################################################################################################
##########################################Run Epoch######################################################
#########################################################################################################

def run_epoch(session, m, words, pos, chunk, pos_vocab_size, chunk_vocab_size,
              verbose=False, valid=False, model_type='JOINT'):
    """Runs the model on the given data."""
    epoch_size = ((len(words) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    comb_loss = 0.0
    pos_total_loss = 0.0
    chunk_total_loss = 0.0
    iters = 0
    accuracy = 0.0
    pos_predictions = []
    pos_true = []
    chunk_predictions = []
    chunk_true = []

    for step, (x, y_pos, y_chunk) in enumerate(create_batches(words, pos, chunk, m.batch_size,
                                                                     m.num_steps, pos_vocab_size, chunk_vocab_size)):

        if model_type == 'POS':
            if valid:
                eval_op = tf.no_op()
            else:
                eval_op = m.pos_op
        elif model_type == 'CHUNK':
            if valid:
                eval_op = tf.no_op()
            else:
                eval_op = m.chunk_op
        else:
            if valid:
                eval_op = tf.no_op()
            else:
                eval_op = m.joint_op

        joint_loss, _, pos_int_pred, chunk_int_pred, pos_int_true, \
        chunk_int_true, pos_loss, chunk_loss = \
            session.run([m.joint_loss, eval_op, m.pos_int_pred,
                         m.chunk_int_pred, m.pos_int_targ, m.chunk_int_targ,
                         m.pos_loss, m.chunk_loss],
                        {m.input_data: x,
                         m.pos_targets: y_pos,
                         m.chunk_targets: y_chunk})
        comb_loss += joint_loss
        chunk_total_loss += chunk_loss
        pos_total_loss += pos_loss
        iters += 1
        if verbose and step % 5 == 0:
            if model_type == 'POS':
                costs = pos_total_loss
                cost = pos_loss
            elif model_type == 'CHUNK':
                costs = chunk_total_loss
                cost = chunk_loss
            else:
                costs = comb_loss
                cost = joint_loss
            print("Type: %s,cost: %3f, total cost: %3f" % (model_type, cost, costs))

        pos_int_pred = np.reshape(pos_int_pred, [m.batch_size, m.num_steps])
        pos_predictions.append(pos_int_pred)
        pos_true.append(pos_int_true)

        chunk_int_pred = np.reshape(chunk_int_pred, [m.batch_size, m.num_steps])
        chunk_predictions.append(chunk_int_pred)
        chunk_true.append(chunk_int_true)

    return (comb_loss / iters), pos_predictions, chunk_predictions, pos_true, \
           chunk_true, (pos_total_loss / iters), (chunk_total_loss / iters)






####################################################################################################
####################################################################################################
##############################Graph#################################################################
####################################################################################################



class Shared_Model(object):
    """Tensorflow Graph For Shared Pos & Chunk Model"""

    def __init__(self, config, is_training):
        self.max_grad_norm = config.max_grad_norm
        self.num_steps = num_steps = config.num_steps
        self.encoder_size = config.encoder_size
        self.pos_decoder_size = config.pos_decoder_size
        self.chunk_decoder_size = config.chunk_decoder_size
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.num_pos_tags = config.num_pos_tags
        self.num_chunk_tags = config.num_chunk_tags
        self.input_data = tf.placeholder(tf.int32, [config.batch_size, num_steps])
        self.word_embedding_size = config.word_embedding_size
        self.pos_embedding_size = config.pos_embedding_size
        self.num_shared_layers = config.num_shared_layers
        self.argmax = config.argmax

        # add input size - size of pos tags
        self.pos_targets = tf.placeholder(tf.float32, [(self.batch_size * num_steps),
                                                       self.num_pos_tags])
        self.chunk_targets = tf.placeholder(tf.float32, [(self.batch_size * num_steps),
                                                         self.num_chunk_tags])

        self._build_graph(config, is_training)

    def _shared_layer(self, input_data, config, is_training):
        """Build the model up until decoding.

        Args:
            input_data = size batch_size X num_steps X embedding size

        Returns:
            output units
        """

        with tf.variable_scope('encoder'):
            lstm_cell = rnn.BasicLSTMCell(config.encoder_size, reuse=tf.get_variable_scope().reuse, forget_bias=1.0)
            if is_training and config.keep_prob < 1:
                lstm_cell = rnn.DropoutWrapper(
                    lstm_cell, output_keep_prob=config.keep_prob)
            encoder_outputs, encoder_states = tf.nn.dynamic_rnn(lstm_cell,
                                                                input_data,
                                                                dtype=tf.float32,
                                                                scope="encoder_rnn")

        return encoder_outputs

    def _pos_private(self, encoder_units, config, is_training):
        """Decode model for pos

        Args:
            encoder_units - these are the encoder units
            num_pos - the number of pos tags there are (output units)

        returns:
            logits
        """
        with tf.variable_scope("pos_decoder"):
            pos_decoder_cell = rnn.BasicLSTMCell(config.pos_decoder_size,
                                     forget_bias=1.0, reuse=tf.get_variable_scope().reuse)

            if is_training and config.keep_prob < 1:
                pos_decoder_cell = rnn.DropoutWrapper(
                    pos_decoder_cell, output_keep_prob=config.keep_prob)

            encoder_units = tf.transpose(encoder_units, [1, 0, 2])
#main layer here
            decoder_outputs, decoder_states = tf.nn.dynamic_rnn(pos_decoder_cell,
                                                                encoder_units,
                                                                dtype=tf.float32,
                                                                scope="pos_rnn")

            output = tf.reshape(tf.concat(decoder_outputs, 1),
                                [-1, config.pos_decoder_size])

            softmax_w = tf.get_variable("softmax_w",
                                        [config.pos_decoder_size,
                                         config.num_pos_tags])
            softmax_b = tf.get_variable("softmax_b", [config.num_pos_tags])
            logits = tf.matmul(output, softmax_w) + softmax_b

        return logits, decoder_states

    def _chunk_private(self, encoder_units, pos_prediction, config, is_training):
        """Decode model for chunks

        Args:
            encoder_units - these are the encoder units:
            [batch_size X encoder_size] with the one the pos prediction
            pos_prediction:
            must be the same size as the encoder_size

        returns:
            logits
        """
        # concatenate the encoder_units and the pos_prediction

        pos_prediction = tf.reshape(pos_prediction,
                                    [self.batch_size, self.num_steps, self.pos_embedding_size])
        encoder_units = tf.transpose(encoder_units, [1, 0, 2])
        chunk_inputs = tf.concat([pos_prediction, encoder_units], 2)

        with tf.variable_scope("chunk_decoder"):
            cell = rnn.BasicLSTMCell(config.chunk_decoder_size, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)

            if is_training and config.keep_prob < 1:
                cell = rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)

            decoder_outputs, decoder_states = tf.nn.dynamic_rnn(cell,
                                                                chunk_inputs,
                                                                dtype=tf.float32,
                                                                scope="chunk_rnn")

            output = tf.reshape(tf.concat(decoder_outputs, 1),
                                [-1, config.chunk_decoder_size])

            softmax_w = tf.get_variable("softmax_w",
                                        [config.chunk_decoder_size,
                                         config.num_chunk_tags])
            softmax_b = tf.get_variable("softmax_b", [config.num_chunk_tags])
            logits = tf.matmul(output, softmax_w) + softmax_b

        return logits, decoder_states

    def _loss(self, logits, labels):
        """Calculate loss for both pos and chunk
            Args:
                logits from the decoder
                labels - one-hot
            returns:
                loss as tensor of type float
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels,
                                                                name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        (_, int_targets) = tf.nn.top_k(labels, 1)
        (_, int_predictions) = tf.nn.top_k(logits, 1)
        num_true = tf.reduce_sum(tf.cast(tf.equal(int_targets, int_predictions), tf.float32))
        accuracy = num_true / (self.num_steps * self.batch_size)
        return loss, accuracy, int_predictions, int_targets

    def _training(self, loss, config):
        """Sets up training ops

        Creates the optimiser

        The op returned from this is what is passed to session run

            Args:
                loss float
                learning_rate float

            returns:

            Op for training
        """
        # Create the gradient descent optimizer with the
        # given learning rate.
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        return train_op

    def _build_graph(self, config, is_training):
        word_embedding = tf.get_variable("word_embedding", [config.vocab_size, config.word_embedding_size])
        inputs = tf.nn.embedding_lookup(word_embedding, self.input_data)
        pos_embedding = tf.get_variable("pos_embedding", [config.num_pos_tags, config.pos_embedding_size])

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        encoding = self._shared_layer(inputs, config, is_training)

        encoding = tf.stack(encoding)
        encoding = tf.transpose(encoding, perm=[1, 0, 2])

        pos_logits, pos_states = self._pos_private(encoding, config, is_training)
        pos_loss, pos_accuracy, pos_int_pred, pos_int_targ = self._loss(pos_logits, self.pos_targets)
        self.pos_loss = pos_loss

        self.pos_int_pred = pos_int_pred
        self.pos_int_targ = pos_int_targ

        # choose either argmax or dot product for pos
        if config.argmax == 1:
            pos_to_chunk_embed = tf.nn.embedding_lookup(pos_embedding, pos_int_pred)
        else:
            pos_to_chunk_embed = tf.matmul(tf.nn.softmax(pos_logits), pos_embedding)

        chunk_logits, chunk_states = self._chunk_private(encoding, pos_to_chunk_embed, config, is_training)
        chunk_loss, chunk_accuracy, chunk_int_pred, chunk_int_targ = self._loss(chunk_logits, self.chunk_targets)
        self.chunk_loss = chunk_loss

        self.chunk_int_pred = chunk_int_pred
        self.chunk_int_targ = chunk_int_targ
        self.joint_loss = chunk_loss + pos_loss

        # return pos embedding
        self.pos_embedding = pos_embedding

        if not is_training:
            return

        self.pos_op = self._training(pos_loss, config)
        self.chunk_op = self._training(chunk_loss, config)
        self.joint_op = self._training(chunk_loss + pos_loss, config)








################################################################################################################################
###############################################Run Model########################################################################
################################################################################################################################
################################################################################################################################

class Config(object):
    """Configuration for the network"""
    init_scale = 0.1  # initialisation scale
    learning_rate = 0.001  # learning_rate (if you are using SGD)
    max_grad_norm = 5  # for gradient clipping
    num_steps = 20  # length of sequence
    word_embedding_size = 400  # size of the embedding
    encoder_size = 200  # first layer
    pos_decoder_size = 200  # second layer
    chunk_decoder_size = 200  # second layer
    max_epoch = 1  # maximum number of epochs
    keep_prob = 0.5  # for dropout
    batch_size = 64  # number of sequence
    vocab_size = 20000  # this isn't used - need to look at this
    num_pos_tags = 45  # hard coded, should it be?
    num_chunk_tags = 23  # as above
    pos_embedding_size = 400
    num_shared_layers = 2
    argmax = 0


def main(model_type, dataset_path, save_path):
    """Main"""
    config = Config()
    raw_data = raw_x_y_data(
        dataset_path, config.num_steps)
    words_t, pos_t, chunk_t, words_v, \
    pos_v, chunk_v, word_to_id, pos_to_id, \
    chunk_to_id, words_test, pos_test, chunk_test, \
    words_c, pos_c, chunk_c = raw_data

    config.num_pos_tags = len(pos_to_id)
    config.num_chunk_tags = len(chunk_to_id)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        # model to train hyperparameters on
        with tf.variable_scope("hyp_model", reuse=None, initializer=initializer):
            m = Shared_Model(is_training=True, config=config)
        with tf.variable_scope("hyp_model", reuse=True, initializer=initializer):
            mvalid = Shared_Model(is_training=False, config=config)

        # model that trains, given hyper-parameters
        with tf.variable_scope("final_model", reuse=None, initializer=initializer):
            mTrain = Shared_Model(is_training=True, config=config)
        with tf.variable_scope("final_model", reuse=True, initializer=initializer):
            mTest = Shared_Model(is_training=False, config=config)

        tf.initialize_all_variables().run()

        # Create an empty array to hold [epoch number, loss]
        best_epoch = [0, 100000]

        print('finding best epoch parameter')
        # ====================================
        # Create vectors for training results
        # ====================================

        # Create empty vectors for loss
        train_loss_stats = np.array([])
        train_pos_loss_stats = np.array([])
        train_chunk_loss_stats = np.array([])
        # Create empty vectors for accuracy
        train_pos_stats = np.array([])
        train_chunk_stats = np.array([])

        # ====================================
        # Create vectors for validation results
        # ====================================
        # Create empty vectors for loss
        valid_loss_stats = np.array([])
        valid_pos_loss_stats = np.array([])
        valid_chunk_loss_stats = np.array([])
        # Create empty vectors for accuracy
        valid_pos_stats = np.array([])
        valid_chunk_stats = np.array([])

        for i in range(config.max_epoch):
            print("Epoch: %d" % (i + 1))
            mean_loss, posp_t, chunkp_t, post_t, chunkt_t, pos_loss, chunk_loss = \
                run_epoch(session, m,
                          words_t, pos_t, chunk_t,
                          config.num_pos_tags, config.num_chunk_tags,
                          verbose=True, model_type=model_type)

            # Save stats for charts
            train_loss_stats = np.append(train_loss_stats, mean_loss)
            train_pos_loss_stats = np.append(train_pos_loss_stats, pos_loss)
            train_chunk_loss_stats = np.append(train_chunk_loss_stats, chunk_loss)

            # get predictions as list
            posp_t = res_to_list(posp_t, config.batch_size, config.num_steps,
                                        pos_to_id, len(words_t))
            chunkp_t = res_to_list(chunkp_t, config.batch_size,
                                          config.num_steps, chunk_to_id, len(words_t))
            post_t = res_to_list(post_t, config.batch_size, config.num_steps,
                                        pos_to_id, len(words_t))
            chunkt_t = res_to_list(chunkt_t, config.batch_size,
                                          config.num_steps, chunk_to_id, len(words_t))

            # find the accuracy
            pos_acc = np.sum(posp_t == post_t) / float(len(posp_t))
            chunk_acc = np.sum(chunkp_t == chunkt_t) / float(len(chunkp_t))

            # add to array
            train_pos_stats = np.append(train_pos_stats, pos_acc)
            train_chunk_stats = np.append(train_chunk_stats, chunk_acc)

            # print for tracking
            print("Pos Training Accuracy After Epoch %d :  %3f" % (i + 1, pos_acc))
            print("Chunk Training Accuracy After Epoch %d : %3f" % (i + 1, chunk_acc))

            valid_loss, posp_v, chunkp_v, post_v, chunkt_v, pos_v_loss, chunk_v_loss = \
                run_epoch(session, mvalid, words_v, pos_v, chunk_v,
                          config.num_pos_tags, config.num_chunk_tags,
                          verbose=True, valid=True, model_type=model_type)

            # Save loss for charts
            valid_loss_stats = np.append(valid_loss_stats, valid_loss)
            valid_pos_loss_stats = np.append(valid_pos_loss_stats, pos_v_loss)
            valid_chunk_loss_stats = np.append(valid_chunk_loss_stats, chunk_v_loss)

            # get predictions as list

            posp_v = res_to_list(posp_v, config.batch_size, config.num_steps,
                                        pos_to_id, len(words_v))
            chunkp_v = res_to_list(chunkp_v, config.batch_size,
                                          config.num_steps, chunk_to_id, len(words_v))
            chunkt_v = res_to_list(chunkt_v, config.batch_size,
                                          config.num_steps, chunk_to_id, len(words_v))
            post_v = res_to_list(post_v, config.batch_size, config.num_steps,
                                        pos_to_id, len(words_v))

            # find accuracy
            pos_acc = np.sum(posp_v == post_v) / float(len(posp_v))
            chunk_acc = np.sum(chunkp_v == chunkt_v) / float(len(chunkp_v))

            print("Pos Validation Accuracy After Epoch %d :  %3f" % (i + 1, pos_acc))
            print("Chunk Validation Accuracy After Epoch %d : %3f" % (i + 1, chunk_acc))

            # add to stats
            valid_pos_stats = np.append(valid_pos_stats, pos_acc)
            valid_chunk_stats = np.append(valid_chunk_stats, chunk_acc)

            # update best parameters
            if (valid_loss < best_epoch[1]):
                best_epoch = [i + 1, valid_loss]

        # Save loss & accuracy plots
        np.savetxt(save_path + '/loss/valid_loss_stats.txt', valid_loss_stats)
        np.savetxt(save_path + '/loss/valid_pos_loss_stats.txt', valid_pos_loss_stats)
        np.savetxt(save_path + '/loss/valid_chunk_loss_stats.txt', valid_chunk_loss_stats)
        np.savetxt(save_path + '/accuracy/valid_pos_stats.txt', valid_pos_stats)
        np.savetxt(save_path + '/accuracy/valid_chunk_stats.txt', valid_chunk_stats)

        np.savetxt(save_path + '/loss/train_loss_stats.txt', train_loss_stats)
        np.savetxt(save_path + '/loss/train_pos_loss_stats.txt', train_pos_loss_stats)
        np.savetxt(save_path + '/loss/train_chunk_loss_stats.txt', train_chunk_loss_stats)
        np.savetxt(save_path + '/accuracy/train_pos_stats.txt', train_pos_stats)
        np.savetxt(save_path + '/accuracy/train_chunk_stats.txt', train_chunk_stats)

        # Train given epoch parameter
        print('Train Given Best Epoch Parameter :' + str(best_epoch[0]))
        for i in range(best_epoch[0]):
            print("Epoch: %d" % (i + 1))
            _, posp_c, chunkp_c, _, _, _, _ = \
                run_epoch(session, mTrain,
                          words_c, pos_c, chunk_c,
                          config.num_pos_tags, config.num_chunk_tags,
                          verbose=True, model_type=model_type)

        print('Getting Testing Predictions')
        _, posp_test, chunkp_test, _, _, _, _ = \
            run_epoch(session, mTest,
                      words_test, pos_test, chunk_test,
                      config.num_pos_tags, config.num_chunk_tags,
                      verbose=True, valid=True, model_type=model_type)

        print('Writing Predictions')
        # prediction reshaping
        posp_c = res_to_list(posp_c, config.batch_size, config.num_steps,
                                    pos_to_id, len(words_c))
        posp_test = res_to_list(posp_test, config.batch_size, config.num_steps,
                                       pos_to_id, len(words_test))
        chunkp_c = res_to_list(chunkp_c, config.batch_size,
                                      config.num_steps, chunk_to_id, len(words_c))
        chunkp_test = res_to_list(chunkp_test, config.batch_size, config.num_steps,
                                         chunk_to_id, len(words_test))

        # save pickle - save_path + '/saved_variables.pkl'
        print('saving variables (pickling)')
        saveload.save(save_path + '/saved_variables.pkl', session)

        train_custom = read_tokens(dataset_path + '/train.txt', 0)
        valid_custom = read_tokens(dataset_path + '/validation.txt', 0)
        combined = read_tokens(dataset_path + '/train_val_combined.txt', 0)
        test_data = read_tokens(dataset_path + '/test.txt', 0)

        print('loaded text')

        chunk_pred_train = np.concatenate((np.transpose(train_custom), chunkp_t), axis=1)
        chunk_pred_val = np.concatenate((np.transpose(valid_custom), chunkp_v), axis=1)
        chunk_pred_c = np.concatenate((np.transpose(combined), chunkp_c), axis=1)
        chunk_pred_test = np.concatenate((np.transpose(test_data), chunkp_test), axis=1)
        pos_pred_train = np.concatenate((np.transpose(train_custom), posp_t), axis=1)
        pos_pred_val = np.concatenate((np.transpose(valid_custom), posp_v), axis=1)
        pos_pred_c = np.concatenate((np.transpose(combined), posp_c), axis=1)
        pos_pred_test = np.concatenate((np.transpose(test_data), posp_test), axis=1)

        print('finished concatenating, about to start saving')

        np.savetxt(save_path + '/predictions/chunk_pred_train.txt',
                   chunk_pred_train, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_train.txt')
        np.savetxt(save_path + '/predictions/chunk_pred_val.txt',
                   chunk_pred_val, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/chunk_pred_combined.txt',
                   chunk_pred_c, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/chunk_pred_test.txt',
                   chunk_pred_test, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/pos_pred_train.txt',
                   pos_pred_train, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/pos_pred_val.txt',
                   pos_pred_val, fmt='%s')
        print('writing to ' + save_path + '/predictions/chunk_pred_val.txt')
        np.savetxt(save_path + '/predictions/pos_pred_combined.txt',
                   pos_pred_c, fmt='%s')
        np.savetxt(save_path + '/predictions/pos_pred_test.txt',
                   pos_pred_test, fmt='%s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type")
    parser.add_argument("--dataset_path")
    parser.add_argument("--save_path")
    args = parser.parse_args()
    if (str(args.model_type) != "POS") and (str(args.model_type) != "CHUNK"):
        args.model_type = 'JOINT'
    print('Model Selected : ' + str(args.model_type))
    main(str(args.model_type), str(args.dataset_path), str(args.save_path))