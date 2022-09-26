# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from fairseq import options, sequence_generator
from fairseq.custom import evaluate_utils
import argparse
from glob import glob
import os.path
import getpass
import sys
import shlex
import pickle
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
matplotlib.style.use('ggplot')
sns.set_style("whitegrid")

REPETITION = 80

# Just randomly select some prefixs
sentence_list = {
    0:{
        'base_sentence': 'His mother was a college softball player for the NEO Lady Norse .',
        'my_sentence': 'His mother was a college softball player for the NEO Lady Norse .' + ' China is super popular .' * REPETITION,
        'keyword': 'super'
    },
    1:{
        'base_sentence': 'Many courts have declined to assess the religious status of Scientology .',
        'my_sentence': 'Many courts have declined to assess the religious status of Scientology .' + ' Apple eat religious finish background .' * REPETITION,
        'keyword': 'religious'
    },
    2: {
        'base_sentence': 'I like to play basketball .',
        'my_sentence': 'I like to play basketball .' + ' basketball play to like player .' * REPETITION,
        'keyword': 'like'
    },
   3: {
        'base_sentence': 'In a post to his blog , Blythe explained that he met the family in private after the trial .',
        'my_sentence': 'In a post to his blog , Blythe explained that he met the family in private after the trial .' + ' Blythe is a good blog .' * REPETITION,
        'keyword': '\.'
    },

}



def main():
    script_parser = argparse.ArgumentParser(description='Computes greedy completion, single-token prediction, and corresponding targets.')
    script_parser.add_argument('--data-dir', type=str, required=True)
    script_parser.add_argument('--base-dir', type=str, required=True)
    script_parser.add_argument('--eval-mode', choices=['all', 'completion', 'singletoken'], default='all')
    script_parser.add_argument('--data-prefix-length', type=int, default=50, help='Length of prefix')
    script_parser.add_argument('--batch-size-completions', type=int, default=128)
    script_parser.add_argument('--batch-size-single-prediction', type=int, default=1024)

    script_parser.add_argument('--completion-length', type=int, default=500,
                               help='The length of each generated sequence, not counting the prefix length')
    script_parser.add_argument('--model-path', type=str, required=True, help='The path to the folder with checkpoints')
    script_parser.add_argument('--save-path', type=str, default='tmp.txt')
    script_parser.add_argument('--save_dir', type=str, default='tmp')
    script_parser.add_argument('--ckpt', choices=['best', 'last', 'all', 'step', 'epoch'], default='best')
    script_parser.add_argument('--ckpt-step', type=str, default=None)
    script_parser.add_argument('--ckpt-epoch', type=str, default=None)
    script_parser.add_argument('--document_path', type=str, default=None)
    script_parser.add_argument('--data-split', choices=['train', 'valid', 'test'], default='valid')
    script_parser.add_argument('--num-samples', type=int, default=-1)
    script_parser.add_argument('--beam-size', type=int, default=1)
    script_parser.add_argument('--batchsize', type=int, default=16)
    script_parser.add_argument('--num_batch', type=int, default=4)
    script_parser.add_argument('--beam-ngram-block', type=int, default=0)
    script_parser.add_argument('--topp', type=float, default=0.0)
    script_parser.add_argument('--completition_nums', type=int, default=200)
    script_parser.add_argument('--random_net_init', action='store_true', default=False)
    script_parser.add_argument('--require_attr', type=int, default=1)
    script_parser.add_argument('--topk', type=int, default=1)
    script_parser.add_argument('--singletoken-topk', type=int, default=1)
    script_parser.add_argument('--singletoken-topp', type=float, default=0.0)


    high_level_args = script_parser.parse_args()

    if high_level_args.ckpt == 'last':
        checkpoints = glob(os.path.join(high_level_args.model_path, 'checkpoint_last.pt'))
    elif high_level_args.ckpt == 'best':
        checkpoints = glob(os.path.join(high_level_args.model_path, 'checkpoint_best.pt'))
    elif high_level_args.ckpt == 'step':
        checkpoints = glob(os.path.join(high_level_args.model_path, 'checkpoint_*_{}.pt'.format(high_level_args.ckpt_step)))
    elif high_level_args.ckpt == 'epoch':
        checkpoints = glob(
            os.path.join(high_level_args.model_path, 'checkpoint{}.pt'.format(high_level_args.ckpt_epoch)))
    elif high_level_args.ckpt == 'all':
        checkpoints = glob(os.path.join(high_level_args.model_path, 'checkpoint*'))

    for i, checkpoint in enumerate(checkpoints):
        if high_level_args.eval_mode in ['all', 'completion']:
            num_tokens = high_level_args.data_prefix_length*high_level_args.batch_size_completions
            FAIRSEQ_OPTS = "--data {} \
                            --task language_modeling_with_generation \
                            --path {} \
                            --tokens-per-sample {} \
                            --max-tokens {} \
                            --sample-break-mode none \
                            --gen-subset {} \
                            --user-dir {}".format(high_level_args.data_dir, checkpoint,
                                                  num_tokens, num_tokens, high_level_args.data_split,
                                                  os.path.join(high_level_args.base_dir, 'fairseq/custom'))
            sys.argv = shlex.split(FAIRSEQ_OPTS)
            parser = options.get_generation_parser()
            args = options.parse_args_and_arch(parser)
            args.add_bos_token = False
            args.skip_invalid_size_inputs_valid_test = False


            task, model, generator, itr, step = evaluate_utils.load(args, random_net_init = high_level_args.random_net_init)

            task.dictionary.eos_index = len(task.dictionary) - 1
            task.dictionary.eos_word = task.dictionary.symbols[-1]


            def single_sentence_eval(fig_num, base_sentence, sentence, keyword, completion_length=100, toppk=1, beam_size=1, topp=0.0, require_attr=False):

                source_tokens = [task.dictionary.index(w) for w in sentence.split()]
                if fig_num % 100 == 0:
                    recovered_sentence = ' '.join([task.dictionary[w] for w in source_tokens])
                    print(f'Processing {seleteced_sentence_id}-th sentences :', recovered_sentence)
                target_sentence, target_probs, predicted_tokens, predicted_max_probs = evaluate_utils.single_token_prob_fetch_para(
                    source_tokens,
                    model,
                    task.target_dictionary, )

                target_sentence_decoded = [task.dictionary[t] for t in target_sentence]
                predicted_tokens_decoded = [task.dictionary[t] for t in predicted_tokens]


                results_dict = {
                    'target_tokens_decoded' : target_sentence_decoded,
                    'target_tokens': target_sentence,
                    'target_probs': target_probs,
                    'predicted_tokens_decoded': predicted_tokens_decoded,
                    'predicted_tokens': predicted_tokens,
                    'predicted_max_probs': predicted_max_probs,
                    'base_sentence': base_sentence,
                    'sentence': sentence
                }

                return results_dict


            with open(high_level_args.document_path, 'r') as f:
                documents = f.readlines()
                for seleteced_sentence_id, rep_sen in enumerate(documents):
                    # Random put a prefix and put repetitive sentences as pseudo repetitive genetations
                    # You can delete it and can also observe the self-reinforcement effect
                    base_sentence = sentence_list[seleteced_sentence_id % len(sentence_list)]['base_sentence']
                    rep_sen = ' ' + rep_sen.strip()
                    my_sentence = base_sentence + rep_sen * 100

                    results_dict = single_sentence_eval(seleteced_sentence_id,
                                                   base_sentence,
                                                   my_sentence,
                                                   'ALL', 50,
                                                   1, 1, 0)
                    with open(os.path.join(high_level_args.save_dir, str(seleteced_sentence_id) + '.teacherforce.pkl'), 'wb') as f:
                        pickle.dump(results_dict, f)




if __name__ == '__main__':
    main()
