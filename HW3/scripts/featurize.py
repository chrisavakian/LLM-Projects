'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
import pdb

NUM_TRAINING_EXAMPLES = 4172
NUM_TEST_EXAMPLES = 1000

BASE_DIR = '../data/'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)

def freq_aa(text, freq):
    return text.count('aa')
def freq_bb(text, freq):
    return text.count('bb')
def freq_cc(text, freq):
    return text.count('cc')
def freq_dd(text, freq):
    return text.count('dd')
def freq_ee(text, freq):
    return text.count('ee')
def freq_ff(text, freq):
    return text.count('ff')
def freq_gg(text, freq):
    return text.count('gg')
def freq_hh(text, freq):
    return text.count('hh')
def freq_ii(text, freq):
    return text.count('ii')
def freq_jj(text, freq):
    return text.count('jj')
def freq_kk(text, freq):
    return text.count('kk')
def freq_ll(text, freq):
    return text.count('ll')
def freq_mm(text, freq):
    return text.count('mm')
def freq_nn(text, freq):
    return text.count('nn')
def freq_oo(text, freq):
    return text.count('oo')
def freq_pp(text, freq):
    return text.count('pp')
def freq_qq(text, freq):
    return text.count('qq')
def freq_rr(text, freq):
    return text.count('rr')
def freq_ss(text, freq):
    return text.count('ss')
def freq_tt(text, freq):
    return text.count('tt')
def freq_uu(text, freq):
    return text.count('uu')
def freq_vv(text, freq):
    return text.count('vv')
def freq_ww(text, freq):
    return text.count('ww')
def freq_xx(text, freq):
    return text.count('xx')
def freq_yy(text, freq):
    return text.count('yy')
def freq_zz(text, freq):
    return text.count('zz')

def freq_hot(text, freq):
    return text.count('hot')
def freq_click(text, freq):
    return text.count('click')
def freq_sex(text, freq):
    return text.count('sex')
def freq_porn(text, freq):
    return text.count('porn')
def freq_viagra(text, freq):
    return text.count('viagra')
def freq_attractive(text, freq):
    return text.count('attractive')

def freq_prize(text, freq):
    return text.count('prize')
def freq_risk(text, freq):
    return text.count('risk')
def freq_save(text, freq):
    return text.count('save')

def freq_free(text, freq):
    return text.count('free')
def freq_urgent(text, freq):
    return text.count('urgent')
def freq_guaranteed(text, freq):
    return text.count('guaranteed')
def freq_win(text, freq):
    return text.count('win')
def freq_cash(text, freq):
    return text.count('cash')
def freq_congratulations(text, freq):
    return text.count('congratulations')
def freq_discount(text, freq):
    return text.count('discount')
def freq_special(text, freq):
    return text.count('special')
def freq_100(text, freq):
    return text.count('100%')

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))

    # --------- Add your own features here ---------
    # Make sure type is int or float
    feature.append(freq_aa(text, freq))
    feature.append(freq_bb(text, freq))
    feature.append(freq_cc(text, freq))
    feature.append(freq_dd(text, freq))
    feature.append(freq_ee(text, freq))
    feature.append(freq_ff(text, freq))
    feature.append(freq_gg(text, freq))
    feature.append(freq_hh(text, freq))
    feature.append(freq_ii(text, freq))
    feature.append(freq_jj(text, freq))
    feature.append(freq_kk(text, freq))
    feature.append(freq_ll(text, freq))
    feature.append(freq_mm(text, freq))
    feature.append(freq_nn(text, freq))
    feature.append(freq_oo(text, freq))
    feature.append(freq_pp(text, freq))
    feature.append(freq_qq(text, freq))
    feature.append(freq_rr(text, freq))
    feature.append(freq_ss(text, freq))
    feature.append(freq_tt(text, freq))
    feature.append(freq_uu(text, freq))
    feature.append(freq_vv(text, freq))
    feature.append(freq_ww(text, freq))
    feature.append(freq_xx(text, freq))
    feature.append(freq_yy(text, freq))
    feature.append(freq_zz(text, freq))
    
    feature.append(freq_prize(text, freq))
    feature.append(freq_risk(text, freq))
    feature.append(freq_save(text, freq))

    feature.append(freq_free(text, freq))
    feature.append(freq_urgent(text, freq))
    feature.append(freq_guaranteed(text, freq))
    feature.append(freq_win(text, freq))
    feature.append(freq_cash(text, freq))
    feature.append(freq_congratulations(text, freq))
    feature.append(freq_discount(text, freq))
    feature.append(freq_special(text, freq))
    feature.append(freq_100(text, freq))

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1)).squeeze()

np.savez(BASE_DIR + 'spam-data.npz', training_data=X, training_labels=Y, test_data=test_design_matrix)
