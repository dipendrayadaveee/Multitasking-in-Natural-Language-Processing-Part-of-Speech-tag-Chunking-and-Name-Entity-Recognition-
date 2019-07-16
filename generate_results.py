import os
from pos_eval import pos_eval

import argparse


def generate_results(path):

    # def = ../../data/current_outcome

    chunk_train = path + 'chunk_pred_train.txt'
    chunk_val = path + 'chunk_pred_val.txt'
    chunk_comb = path + 'chunk_pred_combined.txt'
    chunk_test = path + 'chunk_pred_test.txt'

    ner_train = path + 'ner_pred_train.txt'
    ner_val = path + 'ner_pred_val.txt'
    ner_comb = path + 'ner_pred_combined.txt'
    ner_test = path + 'ner_pred_test.txt'

    pos_train = path + 'pos_pred_train.txt'
    pos_val = path + 'pos_pred_val.txt'
    pos_comb = path + 'pos_pred_combined.txt'
    pos_test = path + 'pos_pred_test.txt'

#for chunk results
    print('generating latex tables - chunk train')
    cmd = 'perl eval.pl -l < ' + chunk_train
    os.system(cmd)

    print('generating latex tables - chunk valid')
    cmd = 'perl eval.pl -l < ' + chunk_val
    os.system(cmd)

    print('generating latex tables - chunk combined')
    cmd = 'perl eval.pl -l < ' + chunk_comb
    os.system(cmd)

    print('generating latex tables - chunk test')
    cmd = 'perl eval.pl -l < ' + chunk_test
    os.system(cmd)

#For NER results
    print('generating latex tables - ner train')
    cmd = 'perl eval.pl -l < ' + ner_train
    os.system(cmd)

    print('generating latex tables - ner valid')
    cmd = 'perl eval.pl -l < ' + ner_val
    os.system(cmd)

    print('generating latex tables - ner combined')
    cmd = 'perl eval.pl -l < ' + ner_comb
    os.system(cmd)

    print('generating latex tables - ner test')
    cmd = 'perl eval.pl -l < ' + ner_test
    os.system(cmd)

#For PoS results
    print('generating accuracy - pos train')
    print(pos_eval(pos_train))

    print('generating accruacy - pos valid')
    print(pos_eval(pos_val))

    print('generating accruacy - pos combined')
    print(pos_eval(pos_comb))

    print('generating accruacy - pos test')
    print(pos_eval(pos_test))

    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    path = args.path
    generate_results(path)
