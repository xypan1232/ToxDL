__author__ = 'jasper.zuallaert'

# This file is called by either our bash script, or manually, to initiate training of a specified model.
# It creates a (fake, placeholder) session to allocate one of the four GPUs
import TestLauncher
import sys
import warnings
import tensorflow as tf

if not sys.warnoptions:
    warnings.simplefilter("ignore")

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
fake_sess_test_to_allocate_gpu = tf.Session(config=config)

TestLauncher.runTest(sys.argv[1])



