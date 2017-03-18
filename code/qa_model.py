from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

# imports added by Ryan
from data_util import get_word2embed_dict, preprocess_sequence_data

logging.basicConfig(level=logging.INFO)



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

    def encode(self, inputs, masks, encoder_state_input,scope,lstm_size):
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
        batch_size = tf.shape(inputs)[0]
        num_words = tf.shape(inputs)[1]  #this should be either questions_max_length or context_max_length

        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # LSTM for encoding the question
        encoded = []
        h = encoder_state_input
        with tf.variable_scope(scope):
            for word_step in range(num_words):
                if word_step >= 1:
                    tf.get_variable_scope().reuse_variables()
                output, h = lstm(inputs[:,word_step],h, scope = scope)*masks[:,word_setup]  
                # apply dropout
                output = tf.nn.dropout(output, self.dropout_placeholder)
                encoded.append(output)

#                 # if we want different lstm_size and hidden_size, conversion layer will need to be here
#                 # (along with initializing w and b tf.variables outside of this for loop)
#                 #  just keep lstm_size = hidden_size for now, too complicated
#                 logits = tf.matmul(output, W) + b
#                 output = tf.nn.softmax(logits) # necessary? 
        return (encoded, h)

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
        # THIS JUST CONFUSES ME NOW
        # ----------------------------------------------------
        batch_size = tf.shape(knowledge_rep)[0]
        passage_size = tf.shape(knowledge_rep)[0]
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

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
                logits = tf.matmul(output, softmax_w) + softmax_b)
                # is it the logits we want? 
                decoded.append(logits)

        return (decoded_probability)

    def decode_simple(self, knowledge_rep, scope, lstm_size,n_classes):
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

        batch_size = tf.shape(knowledge_rep)[0]
        passage_size = tf.shape(knowledge_rep)[0]
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)

        # LSTM for decoded_start
        decoded_probability = []
        h = tf.zeros(shape = [batch_size, lstm_size], dtype = tf.float32)
        with tf.variable_scope(scope):
            #setup variables for this scope
            softmax_w = tf.get_variable("softmax_w", 
                            shape = [2*lstm_size,n_classes],
                            initializer = tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("softmax_b", tf.zeros(n_classes), dtype = tf.float32)

            # make LSTM for this scope
            for time_step in range(self.context_max_length):
                output, h = lstm(knowledge_rep[:,self.question_max_length +time_step], h, scope = scope)
                logits = tf.matmul(output, softmax_w) + softmax_b)
                decoded_start_probability.append(tf.nn.softmax(logits))

        return (decoded_probability)


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
        self.question_placeholder = tf.placeholder(tf.int32,(None,self.question_max_length))
        self.context_placeholder = tf.placeholder(tf.int32,(None,self.context_max_length))
        self.labels_placeholder = tf.placeholder(tf.int32,(None,self.context_max_length))
        self.labels_placeholder = tf.placeholder(tf.int32,(None,self.context_max_length))
        self.question_mask_placeholder = tf.placeholder(tf.bool,(None,self.question_max_length))
        self.context_mask_placeholder = tf.placeholder(tf.bool,(None,self.question_max_length))

        self.dropout_placeholder = tf.placeholder(tf.float32)
        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :EncoderQ:
        """
        # should we initialize this as zeros?? 
        h = tf.zeros(shape = [tf.shape(self.question_placeholder)[0], self.lstm_size], dtype = tf.float32)

        # Encode Question Input
        print('question batch size @ setup:',tf.shape(self.question_placeholder)[0])
        assert tf.shape(self.question_placeholder)[1] = self.question_max_length, "Setup System: 'question_placeholder' is of the wrong shape!" 
        print('question mask batch size @ setup:',tf.shape(self.question_mask_placeholder)[0])
        assert tf.shape(self.question_mask_placeholder)[1] = self.question_max_length, "Setup System: 'question_mask_placeholder' is of the wrong shape!"

        encoded_questions, q = self.encoder.encode(inputs = self.question_placeholder,
            masks = self.question_mask_placeholder,
            encoder_state_input = h,
            scope = "LSTM_encode_question",
            lstm_size = self.lstm_size)

        print('encoded_questions batch size @ setup:',tf.shape(encoded_questions)[0])
        assert tf.shape(self.encoded_questions)[1] = self.question_max_length, "Setup System: 'encoded_questions' is of the wrong shape!" 
        print('h batch size @ setup:',tf.shape(h)[0])
        assert tf.shape(self.h)[1] = self.lstm_size, "Setup System: 'h' is of the wrong shape!"

        # Encode Context Input
        print('context batch size @ setup:',tf.shape(self.context_placeholder)[0])
        assert tf.shape(self.context_placeholder)[1] = self.context_max_length, "Setup System: 'context_placeholder' is of the wrong shape!"
        print('context mask batch size @ setup:',tf.shape(self.context_mask_placeholder)[0])
        assert tf.shape(self.context_mask_placeholder)[1] = self.context_max_length, "Setup System: 'context_mask_placeholder' is of the wrong shape!"

        encoded_context, h = Encoder.encode(inputs = self.context_placeholder,
            masks = self.context_mask_placeholder,
            encoder_state_input = q,
            scope = "LSTM_encode_context",
            lstm_size = self.lstm_size)

        print('encoded_context batch size @ setup:',tf.shape(encoded_context)[0])
        assert tf.shape(self.encoded_context)[1] = self.context_max_length, "Setup System: 'encoded_context' is of the wrong shape!" 


        raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            pass

    def optimize(self, session, question_batch, context_batch, answer_batch, question_mask_batch, context_mask_batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        input_feed[self.question_placeholder] = question_batch
        input_feed[self.context_placeholder] = context_batch
        input_feed[self.labels_placeholder] = answer_batch
        input_feed[self.question_mask_placeholder] = question_mask_batch
        input_feed[self.context_mask_placeholder] = context_mask_batch

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

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        input_feed['test_x'] = test_x

        output_feed = [self.setup_system]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

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
            (question_id, context_id, span) = dataset[i]
            (a_s, a_e) = self.answer(session, (question_id, context_id))
            prediction = ' '.join([rev_vocab[context_id[idx]] for idx in range(a_s, a_e + 1)])
            ground_truth = ' '.join([rev_vocab[context_id[idx]] for idx in range(span[0], span[1] + 1)])
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
        embed_dict = get_word2embed_dict(embeddings, vocab)

        train_examples = preprocess_sequence_data(dataset, embed_dict, self.question_max_length, self.context_max_length, self.embedding_size)
        validation_examples = preprocess_sequence_data()

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
            f1, em = self.evaluate_answer(session, dev_set, dev_set_raw)

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
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))








