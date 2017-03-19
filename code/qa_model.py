from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from defs import LBLS
from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul

from lstm_cell import LSTMCell

from evaluate import exact_match_score, f1_score

# imports added by Ryan
from data_util import get_word2embed_dict, preprocess_sequence_data

logging.basicConfig(level=logging.INFO)

def generate_random_hyperparams(lr_min, lr_max, batch_min, batch_max):
    '''generate random learning rate and batch size'''
    # random search through log space for learning rate
    random_learning_rate = 10**np.random.uniform(lr_min, lr_max)
    random_batch_size = np.random.uniform(batch_min, batch_max)
    return random_learning_rate, random_batch_size

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs , masks, dropout, scope, lstm_size, encoder_state_input = None):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        """
        print('\n')
        print(inputs.get_shape())
        print('\n')
        """
        #print(tf.shape(inputs)[0])
        batch_size = tf.shape(inputs)[0]
        passage_length = tf.shape(inputs)[1]
        embedding_size = inputs.get_shape().as_list()[2]

        lstm = LSTMCell(lstm_size=lstm_size)

        # LSTM for encoding the question
        if encoder_state_input != None:
            state = encoder_state_input
        else:
            h = tf.zeros(shape = [batch_size, lstm_size], dtype = tf.float32)
            c = tf.zeros(shape = [batch_size, lstm_size], dtype = tf.float32)
            state = [h,c]

        with tf.variable_scope(scope):
            inpute_size = inputs.get_shape()[1]
            # print(int(inpute_size), type(inpute_size))
            for word_step in range(inputs.get_shape()[1]):
                if word_step >= 1:
                    tf.get_variable_scope().reuse_variables()

                hidden_mask = tf.tile(tf.expand_dims(masks[:,word_step],1), [1,int(inpute_size)])
                output, h = lstm(inputs[:,word_step],state, scope = scope )#*masks[:,word_step]
                """print('\n ~ ~ ~ Output shape' )
                print(output.get_shape())
                print('\n ~ ~ ~ Hidden mask' )
                print(hidden_mask)"""
                print('~ ~ ~  word_step      ',word_step )
                """
                print('Iinputs.get_shape()[1]\n')
                print(inputs.get_shape()[1])
                print(hidden_mask[:,word_step-1])"""

                output = tf.boolean_mask(output,hidden_mask[:,word_step-1],name='boolean_mask')

                # apply dropout
                output = tf.nn.dropout(output, dropout)
                output = tf.reshape(output,[batch_size,1,embedding_size])
                #print('\n ~ ~ ~ Output shape' )
                #print(output.get_shape())
                if word_step == 0:
                    encoded = output
                else:
                #    print('\n ~ ~ ~ ECONDED value (word_step != 0:)')
                #    print(encoded)
                #    print('\n ~ ~ ~ Output value (word_step != 0:)')
                #    print(output)
                    encoded = tf.concat_v2([encoded,output],1)
                # print('\n ~ ~ ~ encoded shape' )
                # print(encoded.get_shape())
        return (encoded, h, state)

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode_lstm(self, knowledge_rep, scope, lstm_size, n_classes):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        # CONCERNS -------------------------------------------
        # ALL h VALUES INITIALIZED TO ZERO FOR NOW
        # THIS JUST CONFUSES ME NOW, don't use this function, use decode_simple
        # ----------------------------------------------------
        batch_size = tf.shape(knowledge_rep)[0]
        passage_size = knowledge_rep.get_shape()[0]
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple = False)

        # LSTM for decoded_start
        decoded_probability = []
        h = tf.zeros(shape = [self.batch_size, lstm_size], dtype = tf.float32)
        with tf.variable_scope(scope):
            #setup variables for this scope
            softmax_w = tf.get_variable("softmax_w",
                            shape = [lstm_size,n_classes],
                            initializer = tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("softmax_b", tf.zeros(self.n_classes), dtype = tf.float32)

            # make LSTM for this scope
            for time_step in range(passage_size):
                output, h = lstm(knowledge_rep[:,self.question_max_length +time_step], h, scope = scope)
                logits = tf.batch_matmul(output, softmax_w) + softmax_b
                # is it the logits we want?
                decoded.append(logits)

        return (decoded_probability)

    def decode_simple(self, question_state, context_words, lstm_size,n_classes):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """

        batch_size = question_state.get_shape()[0]
        context_size = context_words.get_shape()[1]

        # Decoded_start
        decoded_probability = []
        # h = tf.zeros(shape = [batch_size, lstm_size], dtype = tf.float32)
        with tf.variable_scope('SimpleDecoder'):
            #setup variables for this scope
            softmax_w = tf.get_variable("softmax_w",
                            shape = [2*lstm_size,n_classes],
                            initializer = tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable(tf.zeros(n_classes), name="softmax_b", dtype = tf.float32)

            # make predictions for each word
            assert tf.concat_v2(question_state,context_words[:,wordIdx],axis=1).get_shape()[1] == 2*lstm_size, 'Decode_simple: input is not expected shape'
            assert tf.concat_v2(question_state,context_words[:,wordIdx],axis=1).get_shape()[0] == batch_size, 'Decode_simple: input is not expected shape'
            for wordIdx in range(context_size):
                logits = tf.matmul(tf.concat(question_state,context_words[:,wordIdx],axis=1), softmax_w) + softmax_b
                # do we need to apply softmax if we're using cross_entropy soft max?
                decoded_probability.append(tf.nn.softmax(logits))
        assert length(decoded_probability) == context_size, 'Decode_simple: decoded is not expected shape'
        assert decoded_probability[0].get_shape()[0] == batch_size, 'Decode_simple: decoded is not expected shape'
        assert decoded_probability[0].get_shape()[1] == n_classes, 'Decode_simple: decoded is not expected shape'

        return decoded_probability


class QASystem(object):
    # def __init__(self, encoder, decoder, *args):
    def __init__(self, encoder, decoder, FLAGS):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        # CONSTANTS THAT NEED VALUES AND/ OR A HOME
        self.question_max_length = FLAGS.question_size
        self.context_max_length = FLAGS.output_size       # value correct ? I forget
        self.batch_size = FLAGS.batch_size
        # keep self.lstm_size = self.hidden_size for now!!
        self.lstm_size = FLAGS.state_size                # size output by encoding and decoding lstm
        self.hidden_size = FLAGS.state_size              # size of state expected by decoder , state size of match network
        self.n_classes = FLAGS.n_classes                  # classes = [Answer, Not Answer]
        self.lr = FLAGS.learning_rate
        self.dropout = FLAGS.dropout
        self.embedding_size = FLAGS.embedding_size
        self.n_epochs = FLAGS.epochs
        self.report = None
        # ==== set up placeholder tokens ========
        self.encoder = encoder
        self.decoder = decoder
        self.question_placeholder = tf.placeholder(tf.float32,(None,self.question_max_length,self.embedding_size))
        self.context_placeholder = tf.placeholder(tf.float32,(None,self.context_max_length,self.embedding_size))
        self.labels_placeholder = tf.placeholder(tf.float32,(None,self.context_max_length))
        self.question_mask_placeholder = tf.placeholder(tf.bool,(None,self.question_max_length))
        self.context_mask_placeholder = tf.placeholder(tf.bool,(None,self.context_max_length))

        self.dropout_placeholder = tf.placeholder(tf.float32)
        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            # self.setup_embeddings()
            self.setup_system()
            # self.setup_loss()

        # ==== set up training/updating procedure ====
        pass

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        """
        self.pred = self.setup_prediction_op()
        self.loss = self.setup_loss(self.pred)
        self.train_op = self.setup_training_op(self.loss)

    def setup_prediction_op(self):
        # h = tf.zeros(shape = [self.question_placeholder.get_shape()[0], self.lstm_size], dtype = tf.float32)
        # h = tf.zeros(shape = [self.batch_size, self.lstm_size], dtype = tf.float32)

        # Encode Question Input
        # print('question size @ setup:',self.question_placeholder.get_shape())
        assert self.question_placeholder.get_shape()[1] == self.question_max_length, "Setup System: 'question_placeholder' is of the wrong shape!"
        # print('question mask size @ setup:',self.question_mask_placeholder.get_shape()[0])
        assert self.question_mask_placeholder.get_shape()[1] == self.question_max_length, "Setup System: 'question_mask_placeholder' is of the wrong shape!"

        encoded_questions, q, state = self.encoder.encode(inputs = self.question_placeholder,
            masks = self.question_mask_placeholder,
            dropout = self.dropout_placeholder,
            # encoder_state_input = h,
            encoder_state_input = None,
            scope = "LSTM_encode_question",
            lstm_size = self.lstm_size)

        # print('encoded_questions batch size @ setup:',len(encoded_questions))
        # print() encoded_questions[0].get_shape()[0])
        assert encoded_questions[0].get_shape()[0] == self.question_max_length, "Setup System: 'encoded_questions' is of the wrong shape!"
        assert encoded_questions[0].get_shape()[1] == self.embedding_size, "Setup System: 'encoded_questions' is of the wrong shape!"
        #print('h batch size @ setup:',state[0].get_shape()[0])
        assert state[0].get_shape()[1] == self.lstm_size, "Setup System: 'h' is of the wrong shape!"

        # Encode Context Input
        print('context batch size @ setup:',self.context_placeholder.get_shape()[0])
        assert self.context_placeholder.get_shape()[1] == self.context_max_length, "Setup System: 'context_placeholder' is of the wrong shape!"
        print('context mask batch size @ setup:',self.context_mask_placeholder.get_shape()[0])
        assert self.context_mask_placeholder.get_shape()[1] == self.context_max_length, "Setup System: 'context_mask_placeholder' is of the wrong shape!"

        """print('\n self.context_placeholder.get_shape()[0]')
        print(self.context_placeholder.get_shape()[0])
        print('\n')

        #new_shape = self.context_placeholder.get_shape()[0]
        #h = tf.zeros(shape = [new_shape, self.lstm_size], dtype = tf.float32)"""


        #print(self.context_placeholder.get_shape())

        encoded_context, h, state = self.encoder.encode(inputs = self.context_placeholder,
            masks = self.context_mask_placeholder,
            dropout = self.dropout_placeholder,
            encoder_state_input = q,
            scope = "LSTM_encode_context",
            lstm_size = self.lstm_size)

        print('encoded_context batch size @ setup:',encoded_context.get_shape()[0])
        assert encoded_context.get_shape()[1] == self.context_max_length, "Setup System: 'encoded_context' is of the wrong shape!"

        decoded_probability = self.decoder.decode_simple(encoded_questions, encoded_context, self.lstm_size, self.n_classes)

        return decoded_probability

    def setup_loss(self,pred):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            loss = tf.reduce_mean(
                   tf.boolean_mask(
                       tf.nn.sparse_softmax_cross_entropy_with_logits(pred,
                                                                      self.labels_placeholder),
                              self.context_mask_placeholder))
        return loss

    def setup_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return train_op


    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            pass

    def create_feed_dict(self, question_batch, context_batch, question_mask_batch, context_mask_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE (~6-10 lines)
        # feed_dict = {}
        input_feed = {}

        if question_batch is not None:
            input_feed[self.question_placeholder] = question_batch

        if context_batch is not None:
            input_feed[self.context_placeholder] = context_batch

        if answer_batch is not None:
            input_feed[self.labels_placeholder] = answer_batch

        if question_mask_batch is not None:
            input_feed[self.question_mask_placeholder] = question_mask_batch

        if context_mask_batch is not None:
            input_feed[self.context_mask_placeholder] = context_mask_batch

        input_feed[self.dropout_placeholder] = dropout
        # feed_dict[self.dropout_placeholder] = dropout
        ### END YOUR CODE
        return feed_dict

    def optimize(self, session, question_batch, context_batch, answer_batch, question_mask_batch, context_mask_batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """

        input_feed = create_feed_dict(question_batch = question_batch,
                                      context_batch = context_batch,
                                      question_mask_batch = question_mask_batch,
                                      context_mask_batch = context_mask_batch,
                                      labels_batch = labels_batch,
                                      dropout = self.dropout)

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        input_feed['valid_x'] = valid_x
        input_feed['valid_y'] = valid_y


        performance_records = {}

        output_feed = [self.loss]
        Out = session.run(output_feed, input_feed)
        performance_records[(self.lr, self.batch_size)] = Out


        for i in range(10): # random search hyper-parameter space 10 times
            self.lr, self.batch_size = generate_random_hyperparams(1e-5, 1e-1, 5, 50)
            output_feed = [self.loss]
            Out = session.run(output_feed, input_feed)
            performance_records[(self.lr, self.batch_size)] = Out

        self.lr, self.batch_size = min(performance_records, key=performance_records.get)

        output_feed = [self.loss]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, question_batch, context_batch, question_mask_batch, context_mask_batch):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = create_feed_dict(question_batch = question_batch,
                                      context_batch = context_batch,
                                      question_mask_batch = question_mask_batch,
                                      context_mask_batch = context_mask_batch)

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.setup_system]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, *test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost += self.test(sess, valid_x, valid_y)

        return valid_cost

    # def evaluate_answer(self, session, dataset, sample=100, log=False):
    def evaluate_answer(self, session, dataset, vocab, sample=100, log=False):
        # Had to add vocab as an input here for train.py
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        f1 = 0
        em = 0
        total = 0
        data_len = len(dataset)
        for i in np.random.randint(data_len, size = sample):
            (question, context, span, question_mask, context_mask) = dataset[i]
            (a_s, a_e) = self.answer(session, (question, context, question_mask, context_mask))
            prediction = ' '.join([vocab[context[idx]] for idx in range(a_s, a_e + 1)])
            ground_truth = ' '.join([vocab[context[idx]] for idx in range(span[0], span[1] + 1)])
            total += 1
            em += exact_match_score(prediction, ground_truth)
            f1 += f1_score(prediction, ground_truth)

        em = 100.0 * em / total
        f1 = 100.0 * f1 / total

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    # def train(self, session, dataset, train_dir):
    def train(self, session, dataset, embeddings, vocab, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        embeddings = glove word embeddings
        vocab = vocabulary of the word embeddings
        train_dir

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """
        (train, validation) = dataset

        embed_dict = get_word2embed_dict(embeddings, vocab)

        train_examples = preprocess_sequence_data(train, embed_dict, self.question_max_length, self.context_max_length, self.embedding_size)
        validation_examples = preprocess_sequence_data(validation, embed_dict, self.question_max_length, self.context_max_length, self.embedding_size)

        best_score = 0.
        for epoch in range(self.n_epochs):

            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            prog = Progbar(target=1 + int(len(train_examples) / self.batch_size))

            for i, batch in enumerate(minibatches(train_examples, self.batch_size)):
                _, loss = self.optimize(session,*batch)

                prog.update(i + 1, [("train loss", loss)])
                if self.report: self.report.log_train_loss(loss)
            print("")

            logger.info("Evaluating on development data")
            f1, em = self.evaluate_answer(session, validation_examples, vocab)

            if f1 > best_score:
                best_score = f1
                if saver:
                    logger.info("New best score! Saving model in %s", train_dir)
                    saver.save(sess, train_dir)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(t.value().get_shape().eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
