# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from fairseq.custom.metrics import Metrics
import argparse
import pickle
import os
from tqdm import tqdm
from glob import glob
import pandas as pd


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def process_files(files):
    _dict = {}
    for i, filename in tqdm(enumerate(files), total=len(files)):
        key, metrics = process_file(filename)
        if key in _dict:
            _dict[key] = merge_dicts(_dict[key], metrics)
        else:
            _dict[key] = metrics

    return _dict


def process_file(filename):
    filetype = os.path.basename(filename).split('__')[0]
    modelname = os.path.basename(filename).split('__')[1]
    if filetype not in ['metrics', 'completions', 'single_token_predictions', 'targets']:
        raise Exception

    key_dict, metrics = get_metric(filename, filetype)
    key_dict['model_name'] = modelname
    key_tuple = tuple(key_dict.items())
    return key_tuple, metrics


def get_metric(filename, filetype):
    # Get all keys from filename.
    key_value_list = '.'.join(os.path.basename(filename).split('.')[:-1]).split('__')[2:]
    setting_dict = {}
    for kv in key_value_list:
        k_v = kv.split('_')
        key = '_'.join(k_v[:-1])
        val = k_v[-1]
        try:
            _maybe_num = int(val)
        except:
            try:
                _maybe_num = float(val)
            except:
                _maybe_num = val
        setting_dict[key] = _maybe_num

    split = setting_dict['spl']
    prefix_length = setting_dict.get('pfx')
    completion_length = setting_dict.get('cmpl')

    if filetype == 'completions':
        completion_lines = open(filename, 'r').readlines()
        ngram_metrics = Metrics()
        actual_completions = []
        flat_completions = []

        for i, line in enumerate(completion_lines):
            splitted_line = line.split()
            assert len(splitted_line) == (prefix_length + completion_length)
            actual_completions.append(splitted_line[prefix_length:])
            flat_completions.extend(splitted_line[prefix_length:])

        ngram_metrics.update(actual_completions)
        num_unique_tokens_completions = len(set(flat_completions))

        result = merge_dicts(ngram_metrics.report(kind=f'{split}'),
                             {f'{split}/num_uniq_compl': num_unique_tokens_completions})
        # {'test/pct_repeat_1grams': 0.6914, 'test/pct_repeat_2grams': 0.5484,
        # 'test/pct_repeat_3grams': 0.4832, 'test/pct_repeat_4grams': 0.4422,
        # 'test/num_uniq_compl': 10514}

    if filetype == 'targets':
        targets_ngram_metrics = Metrics()
        targets_completions = []  # Slice targets to have same length as completions.
        targets_flat_completions = []
        targets_lines = open(filename, 'r').readlines()

        for line in targets_lines:
            splitted_line = line.split()
            targets_flat_completions.extend(splitted_line)
            segmented_lines = [splitted_line[i * completion_length:i * completion_length + completion_length]
                               for i in range(len(splitted_line) // completion_length)]
            targets_completions.extend(segmented_lines)

        targets_ngram_metrics.update(targets_completions)
        num_unique_target = len(set(targets_flat_completions))

        result = merge_dicts(targets_ngram_metrics.report(kind=f'{split}_human'),
                             {f'{split}/num_uniq_target': num_unique_target})
        # {'valid_human/pct_repeat_1grams': 0.3215, 'valid_human/pct_repeat_2grams': 0.0564,
        # 'valid_human/pct_repeat_3grams': 0.0143, 'valid_human/pct_repeat_4grams': 0.0051,
        # 'valid/num_uniq_target': 18891}

    if filetype == 'single_token_predictions':
        singlepred_flat = []
        single_prediction_lines = open(filename, 'r').readlines()
        for line in single_prediction_lines:
            singlepred_flat.extend(line.split())

        num_unique_singlepred = len(set(singlepred_flat))

        result = {f'{split}/num_uniq_singletok': num_unique_singlepred}
        # {'test/num_uniq_singletok': 11742}
    if filetype == 'metrics':
        result = pickle.load(open(filename, 'rb'))
        # {'valid/target_rank': 450.2195078246327, 'valid/median_target_rank': 1.5070422535211268, 'valid/hits_at_1': 0.3985692362827711,
        # 'valid/hits_at_10': 0.7164018635766336, 'valid/repeat_at_1/16': 0.23441735662497817, 'valid/wrong_repeat_at_1/16': 0.12253843397075986,
        # 'valid/human_repeat_at_1/16': 0.17525706881817263, 'valid/repeat_at_1/32': 0.3876845887358371,
        # 'valid/wrong_repeat_at_1/32': 0.20992345368166657, 'valid/human_repeat_at_1/32': 0.29013627633864164,
        # 'valid/repeat_at_1/128': 0.6145621789511408, 'valid/wrong_repeat_at_1/128': 0.34380140227709216,
        # 'valid/human_repeat_at_1/128': 0.4790255736379258, 'valid/repeat_at_1/512': 0.7291059794344945,
        # 'valid/wrong_repeat_at_1/512': 0.41541769662663225, 'valid/human_repeat_at_1/512': 0.5936750503110556,
        # 'valid/ppl': 24.555871019989443, 'valid/avg_wrong_mass': 0.3947848677635193, 'checkpoint_step': 30808}

    return setting_dict, result


def print_metrics(resulting_dict, mode='pp'):
    if mode == 'pp':
        output = ''
        for k, v in resulting_dict.items():
            if isinstance(v, str):
                output += '{}:\t\t\t{}\n'.format(k, v)
            else:
                output += '{}:\t\t\t{:.{prec}f}\n'.format(k, v, prec=3)
        return output
    elif mode == 'dict':
        return resulting_dict
    elif mode == 'csv':
        keys_list = []
        val_list = []
        for k, v in resulting_dict.items():
            keys_list.append(k)
            if isinstance(v, str):
                val_list.append('{}'.format(v))
            else:
                val_list.append('{:.{prec}f}'.format(v, prec=3))
        return ','.join(keys_list), ','.join(val_list)


def find_tuple(tuple_first_val, key):
    for tup in key:
        if tuple_first_val == tup[0]:
            return tup[1]
    return None


def get_dataframe_for_model(_dict):
    big_list = []
    for key_tuples, value_metrics in _dict.items():
        split = find_tuple('spl', key_tuples)
        beam_size = find_tuple('bms', key_tuples)
        beam_block = find_tuple('bnb', key_tuples)
        model_name = find_tuple('model_name', key_tuples)
        topk = find_tuple('tpk', key_tuples)
        topp = find_tuple('tpp', key_tuples)

        metrics_to_grab = ['checkpoint_step', f'{split}/pct_repeat_1grams', f'{split}/pct_repeat_4grams',
                           f'{split}/num_uniq_compl', f'{split}/ppl', f'{split}/hits_at_1', f'{split}/repeat_at_1/128',
                           f'{split}/wrong_repeat_at_1/128', f'{split}/num_uniq_singletok']
        per_model_list = [model_name, f'{beam_size}', f'{beam_block}', f'{topk}', f'{topp}', f'{split}'] + [
            value_metrics[m] for m in metrics_to_grab]
        big_list.append(per_model_list)

    header = ['model_name', 'beam size', 'beam block', 'topk', 'topp', 'split', 'checkpoint_step', 'seq-rep-1',
              'seq-rep-4', 'uniq-seq', 'ppl', 'acc', 'rep', 'wrep', 'uniq']
    df = pd.DataFrame(big_list, columns=header)

    return df


def main():
    pd.set_option("display.precision", 3)
    parser = argparse.ArgumentParser(description='output postprocessor')

    parser.add_argument('--eval-dirs', nargs='+')
    parser.add_argument('--model-names', nargs='+')

    script_args = parser.parse_args()

    model_to_files = {}

    for model_name in tqdm(script_args.model_names):
        paths = []
        for eval_dir in script_args.eval_dirs:
            paths.extend(glob(os.path.join(eval_dir, '*__{}*__spl_*__*'.format(model_name))))
        model_to_files[model_name] = paths

    model_to_metric_dicts = {}
    for name, list_of_filenames in model_to_files.items():
        model_to_metric_dicts[name] = process_files(list_of_filenames)

    model_to_dataframes = {name: get_dataframe_for_model(_name_dict)
                           for name, _name_dict in model_to_metric_dicts.items()}

    pd.set_option('display.max_columns', 20)
    for mname, df in model_to_dataframes.items():
        print('MODEL: {}'.format(mname) + '\n')
        df = df.sort_values(by=['split', 'checkpoint_step'])
        print(df)
        df.to_csv(f'eval_results_{mname}.csv')
        print('\n\n')


if __name__ == '__main__':
    main()
