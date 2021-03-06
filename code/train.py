from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import numpy as np

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
# Added by Ryan
from six.moves import xrange

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
# tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 300, "The output size of your model.")
# tf.app.flags.DEFINE_integer("output_size", 17, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "../../data/squad/", "SQuAD directory (default ../../data/squad)")
tf.app.flags.DEFINE_string("train_dir", "../../data/squad/", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "../../data/squad", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("val_dir", "../../data/squad", "Validation directory to save the model parameters (default: ./val).")
tf.app.flags.DEFINE_string("load_val_dir", "../../data/squad", "Validation directory to load model parameters from to resume training (default: {val_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "../../data/squad/vocab.dat", "Path to vocab file (default: ../../data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "../../data/squad/glove.trimmed.100.npz", "Path to the trimmed GLoVe embedding (default: ../../data/squad/glove.trimmed.{embedding_size}.npz)")
# user made flags
tf.app.flags.DEFINE_string("embed_type", "glove", "Type of embedding used (default: glove)")
# tf.app.flags.DEFINE_string("question_size", 70, "Size of question (default: 70)")
tf.app.flags.DEFINE_string("question_size", 30, "Size of question (default: 70)")

tf.app.flags.DEFINE_string("n_classes", 3, "Number of output classes (default: 2)")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    # initialize model with specific parameters
    # generate fresh model parameters if no saved model is provided
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    # create dictionary mapping tokens to IDs
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        # grab the list of all words in the vocab
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        # create a dictionary of the tuples of each word and its position in the list
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def get_normalized_val_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = None

    # read training data

    print('##########  READ TRAINING DATA ########## \n')
    context = open(FLAGS.data_dir + 'train.context').read().split('\n')
    question = open(FLAGS.data_dir + 'train.question').read().split('\n')
    answer_span = open(FLAGS.data_dir + 'train.span').read().split('\n')
    train = []
    for k in range(len(context)-1):
        ans_intList = [int(value) for value in answer_span[k].split(' ')]
        train.append((question[k].split(' '),context[k].split(' '),ans_intList))
    questionLenList = [len(q) for q in question]
    meanQuestLen = sum(questionLenList)/len(questionLenList)
    #print("AVERAGE QUESTION LENGTH = ", meanQuestLen)
    contextLenList = [len(c) for c in context]
    meanCLen = sum(contextLenList)/len(contextLenList)
    #print("AVERAGE CONTEXT LENGTH = ", meanCLen)

    # for k in xrange(len(context)):
        # L = [map(int,question[k].split())]
        # L.append(map(int, context[k].split()))
        # L.append(map(int, answer_span[k].split()))
        # train.append((L))

    # read test data
    print('##########  READ TEST DATA ########## \n')
    context = open(FLAGS.data_dir + 'val.context').read().split('\n')
    question = open(FLAGS.data_dir + 'val.question').read().split('\n')
    answer_span = open(FLAGS.data_dir + 'val.span').read().split('\n')
    val = []
    for k in range(len(context)-1):
        ans_intList = [int(value) for value in answer_span[k].split(' ')]
        val.append((question[k].split(' '),context[k].split(' '),ans_intList))

    # for k in xrange(len(context)):
    #     L = [map(int,question[k].split())]
    #     L.append(map(int, context[k].split()))
    #     L.append(map(int, answer_span[k].split()))
    #     val.append((L))


    dataset = (train, val)

    # read word embeddings
    print('##########  READ WORD EMBEDDINGS ########## \n')
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    # We will also load the embeddings here
    embeddings = np.load(embed_path)[FLAGS.embed_type]

    #read word vocabularies
    print('##########  READ WORD VOCABULARIES ########## \n')
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    # initialize encoder, decoder
    print('##########  INITIALIZE ENCODER / DECODER ########## \n')
    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    # create QA model
    print('########## CREATE MODEL    ########## \n')
    # qa = QASystem(encoder, decoder)
    qa = QASystem(encoder, decoder, FLAGS)

    # make log file
    print('##########  MAKE LOG  FILES   ######################\n')

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    # save flags
    print('##########  SAVE FLAGS   ######################\n')
    # print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # run training on QA model on training data
    print('\n ##########    START TRAINING   ############## \n')
    with tf.Session() as sess:
        # load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        load_train_dir = FLAGS.train_dir
        initialize_model(sess, qa, load_train_dir)

        # save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        save_train_dir = FLAGS.train_dir
        # qa.train(sess, dataset, save_train_dir)
        qa.train(sess, dataset, embeddings, vocab, save_train_dir)

        # qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)
        # qa.evaluate_answer(sess, dataset, val, log=True)


        #load_val_dir = get_normalized_train_dir(FLAGS.load_val_dir or FLAGS.val_dir)
        #qa.validate( sess, load_val_dir)

if __name__ == "__main__":
    tf.app.run()
