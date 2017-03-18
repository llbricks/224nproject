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

logging.basicConfig(level=logging.INFO)

# CONSTANTS THAT NEED VALUES AND/ OR A HOME
self.question_max_length = 70
self.context_max_length = 500       # value correct ? I forget
self.batch_size = 32
# keep self.lstm_size = self.hidden_size for now!!
self.lstm_size = 300                # size output by encoding and decoding lstm
self.hidden_size = 300              # size of state expected by decoder , state size of match network
self.n_classes = 2                  # classes = [Answer, Not Answer]
self.lr = 0.001
dropout = 0.5

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def generate_random_hyperparams(lr_min, lr_max, batch_min, batch_max):
    '''generate random learning rate and batch size'''
    # random search through log space for learning rate
    random_learning_rate = 10**np.random.uniform(lr_min, lr_max)
    random_batch_size = np.random.uniform(batch_min, batch_max)
    return random_learning_rate, random_batch_size


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input):
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
        # CONCERNS -------------------------------------------
        # NO MASKING IMPLEMENTED YET!
        # ALL h VALUES INITIALIZED TO ZERO FOR NOW
        # HOW DO I USE ENCODER_STATE_INPUT?
        # ----------------------------------------------------

        # Sanity Check -- I only check the first element
        assert inputs[0][0].shape()[0] == self.batch_size, "Encoder: 'input' is of the wrong shape!"
        assert inputs[0][0].shape()[1] == question_max_length, "Encoder: 'input' is of the wrong shape!"
        assert inputs[0][1].shape()[0] == self.batch_size, "Encoder: 'input' is of the wrong shape!"
        assert inputs[0][1].shape()[1] == context_max_length, "Encoder: 'input' is of the wrong shape!"

        # assert inputs[0][0].shape()[1] == self.embedding_size, "Encoder: 'input' is of the wrong shape!"

        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)


        # LSTM for encoding the question
        scope = "LSTM_encode_question"
        question_encoded = []
        # h_q = tf.zeros(shape = [tf.shape(inputs[0])[0], self.lstm_size], dtype = tf.float32)
        h = tf.zeros(shape = [self.batch_size, self.lstm_size], dtype = tf.float32)
        with tf.variable_scope(scope):
            for time_step in range(self.question_max_length):
                if time_step >= 1:
                    tf.get_variable_scope().reuse_variables()
                output, h = lstm(inputs[0][0][:,time_step],h, scope = scope)
                # apply dropout
                output = tf.nn.dropout(output, self.dropout_placeholder)
                question_encoded.append(output)

#                 # if we want different lstm_size and hidden_size, conversion layer will need to be here
#                 # (along with initializing w and b tf.variables outside of this for loop)
#                 #  just keep lstm_size = hidden_size for now, too complicated
#                 logits = tf.matmul(output, W) + b
#                 output = tf.nn.softmax(logits) # necessary?

        # Sanity Check -- I only check the first element
        assert question_encoded.shape()[0] == self.question_max_length, "Encoder: 'question_encoded' is of the wrong shape!"
        assert question_encoded[0].shape()[0] == self.batch_size, "Encoder: 'question_encoded' is of the wrong shape!"
        assert question_encoded[0].shape()[1] == self.lstm_size, "Encoder: 'question_encoded' is of the wrong shape!"

        # LSTM for encoding the context
        scope = "LSTM_encode_context"
        context_encoded = []
        h = tf.zeros(shape = [self.batch_size, self.lstm_size], dtype = tf.float32)
        with tf.variable_scope(scope):
            for time_step in range(self.context_max_length):
                if time_step >= 1:
                    tf.get_variable_scope().reuse_variables()
                output, h = lstm(inputs[0][1][:,time_step],h, scope = scope)
                # apply dropout
                output = tf.nn.dropout(output, self.dropout_placeholder)
                context_encoded.append(output)

#                 # if we want different lstm_size and hidden_size, conversion layer will need to be here, as in question


        # Sanity Check -- I only check the first element
        assert context_encoded.shape()[0] == self.context_max_length, "Encoder: 'context_encoded' is of the wrong shape!"
        assert context_encoded[0].shape()[0] == self.batch_size, "Encoder: 'context_encoded' is of the wrong shape!"
        assert context_encoded[0].shape()[1] == self.lstm_size, "Encoder: 'context_encoded' is of the wrong shape!"

        return ((questions_encoded,context_encoded), h)

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
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
        # NO MASKING IMPLEMENTED HERE -- IS THAT OK? -- I think so , that's fine.
        # ALL h VALUES INITIALIZED TO ZERO FOR NOW
        # HOW DO I USE ENCODER_STATE_INPUT?
        # NOT DONE YET! HOW DO I SEPARATE THE INPUT ? (question from context)
        # ----------------------------------------------------
        # Sanity Check -- I only check the first element
        assert knowledge_rep[0].shape()[0] == self.batch_size, "Decoder: 'knowledge_rep' is of the wrong shape!"
        assert knowledge_rep[0].shape()[1] == self.question_max_length + self.context_max_length, "Decoder: 'knowledge_rep' is of the wrong shape!"

        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)

        # LSTM for decoded_start
        scope = "LSTM_decode_start"
        decoded_start_probability = []
        h = tf.zeros(shape = [self.batch_size, self.lstm_size], dtype = tf.float32)
        with tf.variable_scope(scope):
            #setup variables for this scope
            softmax_w = tf.get_variable("softmax_w",
                            shape = [self.lstm_size,self.n_classes],
                            initializer = tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("softmax_b", tf.zeros(self.n_classes), dtype = tf.float32)

            # make LSTM for this scope
            for time_step in range(self.context_max_length):
                output, h = lstm(knowledge_rep[:,self.question_max_length +time_step], h, scope = scope)
                logits = tf.matmul(output, softmax_w) + softmax_b)
                decoded_start_probability.append(tf.nn.softmax(logits))



        # LSTM for decoded_end
        scope = "LSTM_decode_end"
        decoded_end_probability = []
        h = tf.zeros(shape = [self.batch_size, self.lstm_size], dtype = tf.float32)
        with tf.variable_scope(scope):
            #setup variables for this scope
            softmax_w = tf.get_variable("softmax_w",
                            shape = [self.lstm_size,self.n_classes],
                            initializer = tf.contrib.layers.xavier_initializer())
            softmax_b = tf.get_variable("softmax_b", tf.zeros(self.config.n_classes), dtype = tf.float32)

            # make LSTM for this scope
            for time_step in range(self.context_max_length):
                if time_step >= 1:
                    tf.get_variable_scope().reuse_variables()
                output, h = lstm(knowledge_rep[:,self.question_max_length +time_step], h, scope = scope)
                logits = tf.matmul(output, softmax_w) + softmax_b)
                decoded_end_probability.append(tf.nn.softmax(logits))

        return (decoded_start_probability,decoded_end_probability)

class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========


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
        return = Encoder(
            tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        lstm_encodec = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)

        encoded_q = Encoder.encode(self.question_placeholder)
        encoded_c = Encoder.encode(self.context_placeholder)

        out = lstm_homie_whatshisface([encoded_q + encoded_c])
        decoded = lstm_homie_whatshisface(out)

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

    def optimize(self, session, train_x, train_y, masks):
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

        performance_records = {}

        for i in range(10): # random search hyper-parameter space 10 times
            self.lr, self.batch_size = generate_random_hyperparams(1e-5, 1e-1, 5, 50)
            output_feed = [self.loss]
            Out = session.run(output_feed, input_feed)
            performance_records[(self.lr, self.batch_size)] = Out


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

        train_examples = preprocess_sequence_data(dataset, embed_dict)

        best_score = 0.
        for epoch in range(self.config.n_epochs):

            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            prog = Progbar(target=1 + int(len(train_examples) / self.batch_size))

            for i, batch in enumerate(minibatches(train_examples, self.batch_size)):

                loss = self.optimize(session,*batch)

                prog.update(i + 1, [("train loss", loss)])
                if self.report: self.report.log_train_loss(loss)
            print("")

            logger.info("Evaluating on development data")
            f1, em = self.evaluate_answer(sess, dev_set, dev_set_raw)

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








def get_word2embed_dict(embedding, vocab):
    """
    Load word vector mapping using @embedding, @vocab.
    Assumes each line of the vocab file matches with those of the embedding
    file.
    """
    ret = OrderedDict()
    for key in vocab.keys():
        ret[key] = embedding[vocab[key]]

    return ret


def preprocess_sequence_data(dataset, embed_dict):

    ret = []
    for ((question, context), answer_span) in dataset:
        # replace tokens with corresponding embedding
        question_embed = embed(question, embed_dict)
        context_embed = embed(context, embed_dict)
        # create list of labels of max_length
        answer_labels = labelize(answer_span)
        # pad question and context to be max_length
        # also return masks for BOTH question and context
        (question_data, context_data), masks = pad(question_embed, context_embed)

        ret.append(((question_data, context_data), answer_labels, masks))

    return ret

def embed(tokens, embed_dict):

    ret = []
    for token in tokens:
        # normalize token to find it in embed_dict
        word = normalize(token)
        # word's embedding (UNK's embedding otherwise)
        wv = embed_dict.get(word, embed_dict[UNK])

        ret.append(wv)

    return ret

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit():
        return NUM
    else:
        return word.lower()

def labelize(span):
    # create negative label list
    labels = LBLS[-1]*self.context_max_length
    # set appropriate labels positive
    labels[span[0]:span[1] + 1] = LBLS[0] * (span[1] - span[0] + 1)
    return labels

def pad(question, context):
    # initialize padding variables
    zero_vector = [0] * Config.n_features
    zero_label = 0
    # pad question to question_max_length
    pad_len = max(self.question_max_length - len(question), 0)
    padding = zero_vector * pad_len
    question = question + padding
    question_in = question[:self.question_max_length]
    question_mask = [True] * (self.question_max_length - pad_len)
                    + [False] * pad_len
    # pad context to context_max_length
    pad_len = max(self.context_max_length - len(context), 0)
    padding = zero_vector * pad_len
    context = question + padding
    context_in = question[:self.context_max_length]
    context_mask = [True] * (self.context_max_length - pad_len)
                    + [False] * pad_len

    return (question_in, context_in), (question_mask, context_mask)
