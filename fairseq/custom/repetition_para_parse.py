import glob
import os
import pickle
import re
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

class AvgMetric():
    def __init__(self):
        self.data = []
        self.cnt = 0

    def add(self, d, cnt):
        self.data.append(d)
        self.cnt += cnt

    def mean(self):
        return sum(self.data) / self.cnt

def position_study(file_dir):
    all_data_dict = {}
    for pkl in glob.glob(os.path.join(file_dir, '*.teacherforce.pkl')):

        # 1 avg first prob
        # 2 avg last prob
        # 3 avg improve rate
        # 4 avg win rate

        # results_dict = {
        #     'target_tokens_decoded': target_sentence_decoded,
        #     'target_tokens': target_sentence,
        #     'target_probs': target_probs,
        #     'predicted_tokens_decoded': predicted_tokens_decoded,
        #     'predicted_tokens': predicted_tokens,
        #     'predicted_max_probs': predicted_max_probs,
        # }
        results_dict = pickle.load(open(pkl, 'rb'))
        sentence = results_dict['sentence']
        prefix_len = len(results_dict['base_sentence'].split())
        sentence = sentence.split()
        for i in list(range(len(sentence)))[:-1][::-1]:
            if sentence[i] == '.':
                repear_sen = sentence[i+1:]
                break

        assert len(sentence) == len(results_dict['target_tokens_decoded'])

        rep_dict = {}

        rep_len = len(repear_sen)
        first_time_target_prob = results_dict['target_probs'][prefix_len: prefix_len + rep_len]
        first_time_target_tokens = results_dict['target_tokens'][prefix_len: prefix_len + rep_len]
        first_time_max_tokens = results_dict['predicted_tokens'][prefix_len: prefix_len + rep_len]
        # results_dict['target_tokens_decoded'][prefix_len: prefix_len + rep_len]

        repeat_start_pos = prefix_len + len(repear_sen)
        # results_dict['target_tokens_decoded'][start_pos: end_pos] for debug
        for rep_num in range(1, len(sentence[repeat_start_pos:]) // rep_len):
            start_pos = repeat_start_pos + (rep_num - 1) * rep_len
            end_pos = repeat_start_pos + rep_num * rep_len
            repeat_time_target_prob = results_dict['target_probs'][start_pos: end_pos]
            repeat_time_target_tokens = results_dict['target_tokens'][start_pos: end_pos]
            repeat_time_max_tokens = results_dict['predicted_tokens'][start_pos: end_pos]

            # Eos
            # try:
            eos_prob = repeat_time_target_prob[-2]
            # except:
            #     import pdb
            #     pdb.set_trace( )
            #     a = 1
            eos_improve = int(eos_prob>first_time_target_prob[-2])
            eos_win = int(repeat_time_target_tokens[-2] == repeat_time_max_tokens[-2])

            # After Eos
            first_prob = repeat_time_target_prob[-1]
            first_improve = int(first_prob > first_time_target_prob[-1])
            first_win = int(repeat_time_target_tokens[-1] == repeat_time_max_tokens[-1])

            # Others
            sum_last_prob = np.sum(repeat_time_target_prob[:-2])
            improve_count = 0
            win_count = 0
            for i in range(len(repeat_time_target_prob) - 2):
                if repeat_time_target_prob[i] > first_time_target_prob[i]:
                    improve_count += 1
                if repeat_time_target_tokens[i] == repeat_time_max_tokens[i]:
                    win_count += 1
            rep_dict[rep_num] = {
                'sum_last_prob': sum_last_prob,
                'improve_count': improve_count,
                'win_count': win_count,
                'middle_length': len(repeat_time_target_prob) - 2,
                'eos_prob' : eos_prob,
                'eos_improve': eos_improve,
                'eos_win': eos_win,
                'first_prob': first_prob,
                'first_improve': first_improve,
                'first_win' : first_win
            }

        rep_dict['first_sentence'] = {
            'sum_last_prob': sum(first_time_target_prob[:-2]),
            'eos_prob': first_time_target_prob[-2],
            'first_prob': first_time_target_prob[-1],

        }

        all_data_dict[pkl] = rep_dict

    rep_metric_dict = {}

    for target_repetition in range(1, 99):
        first_sum_prob = AvgMetric()
        first_eos_prob = AvgMetric()
        first_first_prob = AvgMetric()

        rep_sum_prob = AvgMetric()
        improve_rate = AvgMetric()

        eos_prob = AvgMetric()
        eos_improve = AvgMetric()
        eos_win = AvgMetric()

        first_prob = AvgMetric()
        first_improve = AvgMetric()
        first_win = AvgMetric()

        win_rate = AvgMetric()
        for pkl, rep_dict in all_data_dict.items():
            length = rep_dict[target_repetition]['middle_length']
            first_sum_prob.add(rep_dict['first_sentence']['sum_last_prob'], length)
            first_eos_prob.add(rep_dict['first_sentence']['eos_prob'], 1)
            first_first_prob.add(rep_dict['first_sentence']['first_prob'], 1)


            rep_sum_prob.add(rep_dict[target_repetition]['sum_last_prob'], length)
            improve_rate.add(rep_dict[target_repetition]['improve_count'], length)
            win_rate.add(rep_dict[target_repetition]['win_count'], length)

            eos_prob.add(rep_dict[target_repetition]['eos_prob'], 1)
            eos_improve.add(rep_dict[target_repetition]['eos_improve'], 1)
            eos_win.add(rep_dict[target_repetition]['eos_win'], 1)

            first_prob.add(rep_dict[target_repetition]['first_prob'], 1)
            first_improve.add(rep_dict[target_repetition]['first_improve'], 1)
            first_win.add(rep_dict[target_repetition]['first_win'], 1)

        rep_metric_dict[target_repetition] = {
            'Middle_AVG_Prob_in_first': first_sum_prob.mean(),
            'Middle_Repeat_AVG_Prob': rep_sum_prob.mean(),
            'Middle_Improve_Rate': improve_rate.mean(),
            'Middle_Win_Rate': win_rate.mean(),
            'First_prob_in_first': first_first_prob.mean(),
            'First_Repeat_prob': first_prob.mean(),
            'First_improve_rate': first_improve.mean(),
            'First_win_rate': first_win.mean(),
            'Eos_prob_in_first': first_eos_prob.mean(),
            'Eos_Repeat_prob': eos_prob.mean(),
            'Eos_improve_rate': eos_improve.mean(),
            'Eos_win_rate': eos_win.mean(),
            'Repetition': target_repetition
        }

    return rep_metric_dict

def draw_rep_figure(results_dict, fig_num, save_dir):
    sentence = results_dict['sentence']
    repear_sen = sentence[list(re.finditer('\.', sentence))[-2].end():].lstrip().split()
    prefix_len = len(results_dict['base_sentence'].split())
    sentence = sentence.split()


    rep_len = len(repear_sen)

    for key_index in range(rep_len):
        # process_core_logic

        ALL_target_prob = []
        ALL_target_token = []
        ALL_max_prob = []
        ALL_max_token = []

        repeat_start_pos = prefix_len + key_index
        for rep_num in range(0, len(sentence[repeat_start_pos:]) // rep_len):
            pos = repeat_start_pos + rep_num * rep_len
            ALL_target_prob.append(results_dict['target_probs'][pos])
            ALL_target_token.append(results_dict['target_tokens_decoded'][pos])
            ALL_max_prob.append(results_dict['predicted_max_probs'][pos])
            ALL_max_token.append(results_dict['predicted_tokens_decoded'][pos])

        initial_prob = ALL_target_prob[0]
        prob_1 = ALL_target_prob[1]
        last_prob = np.mean(ALL_target_prob[-10])
        prob_c = f"From {str(format(initial_prob, '.4e'))} -> {str(format(prob_1, '.4e'))} -> {str(format(last_prob, '.4e'))}"

        def process_for_fit_caption(title_line, prob_c):
            i = 0
            new_line = []
            while (i < len(title_line)):
                i += 100
                new_line.append(title_line[i - 100:i] + '\n')
            new_line.append(prob_c)
            return ''.join(new_line)

        fig = plt.figure(int(key_index), figsize=(30, 10))
        x = range(len(ALL_target_prob))
        bar1 = plt.bar([i - 0.2 for i in x], height=ALL_target_prob, width=0.4,
                       alpha=0.8, color='r', label='Target Token')
        bar2 = plt.bar([i + 0.2 for i in x], height=ALL_max_prob, width=0.4,
                       alpha=0.8, color='b', label='Max Token')
        plt.legend()  # 设置图例
        for i, rect in enumerate(bar1):
            height = rect.get_height()  # 获得bar1的高度
            plt.text(rect.get_x() + rect.get_width() / 2, height + 0.001, str(ALL_target_token[i]), ha="center",
                     va="bottom", rotation=-10)
        for i, rect in enumerate(bar2):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height + 0.001, str(ALL_max_token[i]), ha="center",
                     va="bottom", rotation=-10)

        plt.title(process_for_fit_caption(' '.join(sentence), prob_c), fontsize=10)

        os.makedirs(high_level_args.save_dir, exist_ok=True)

        fig.savefig(f'{save_dir}/ID{fig_num}_key{key_index}.png', dpi=150,
                    bbox_inches='tight')


def main(file_dir, draw_figures=0):
    all_data_dict = {}
    drawed_figures = 0
    for idx, pkl in enumerate(glob.glob(os.path.join(file_dir, '*.teacherforce.pkl'))):
        if idx % 100 == 0:
            print(f'Processing {pkl}')

        # 1 avg first prob
        # 2 avg last prob
        # 3 avg improve rate
        # 4 avg win rate

        # results_dict = {
        #     'target_tokens_decoded': target_sentence_decoded,
        #     'target_tokens': target_sentence,
        #     'target_probs': target_probs,
        #     'predicted_tokens_decoded': predicted_tokens_decoded,
        #     'predicted_tokens': predicted_tokens,
        #     'predicted_max_probs': predicted_max_probs,
        # }
        results_dict = pickle.load(open(pkl, 'rb'))
        sentence = results_dict['sentence']
        repear_sen = sentence[list(re.finditer('\.', sentence))[-2].end():].lstrip().split()
        prefix_len = len(results_dict['base_sentence'].split())
        sentence = sentence.split()
        if drawed_figures < draw_figures:
            draw_rep_figure(results_dict, os.path.split(pkl)[1].split('.')[0],  high_level_args.save_dir)
            drawed_figures += 1

        assert len(sentence) == len(results_dict['target_tokens_decoded'])

        rep_dict = {}


        rep_len = len(repear_sen)
        first_time_target_prob = results_dict['target_probs'][prefix_len: prefix_len+rep_len]
        first_time_target_tokens = results_dict['target_tokens'][prefix_len: prefix_len+rep_len]
        first_time_max_tokens = results_dict['predicted_tokens'][prefix_len: prefix_len+rep_len]

        repeat_start_pos = prefix_len+len(repear_sen)
        # results_dict['target_tokens_decoded'][start_pos: end_pos] for debug
        for rep_num in range(1, len(sentence[repeat_start_pos:])//rep_len):
            start_pos = repeat_start_pos + (rep_num-1)*rep_len
            end_pos = repeat_start_pos + rep_num*rep_len
            repeat_time_target_prob = results_dict['target_probs'][start_pos: end_pos]
            repeat_time_target_tokens = results_dict['target_tokens'][start_pos: end_pos]
            repeat_time_max_tokens = results_dict['predicted_tokens'][start_pos: end_pos]

            sum_last_prob = np.sum(repeat_time_target_prob)
            improve_count = 0
            win_count = 0
            for i in range(len(repeat_time_target_prob)):
                if repeat_time_target_prob[i] > first_time_target_prob[i]:
                    improve_count += 1
                if repeat_time_target_tokens[i] == repeat_time_max_tokens[i]:
                    win_count += 1
            rep_dict[rep_num] = {
                'sum_last_prob': sum_last_prob,
                'improve_count': improve_count,
                'win_count': win_count,
                'length': len(repeat_time_target_prob)
            }

        all_repetition = max(list(rep_dict.keys()))
        rep_dict['first_sentence'] = {
            'sum_last_prob': sum(first_time_target_prob),

        }

        all_data_dict[pkl] = rep_dict

    rep_metric_dict = {}

    for target_repetition in range(1, 99):
        first_sum_prob = AvgMetric()
        rep_sum_prob = AvgMetric()
        improve_rate = AvgMetric()



        win_rate = AvgMetric()
        for pkl,  rep_dict in all_data_dict.items():
            length = rep_dict[target_repetition]['length']
            first_sum_prob.add(rep_dict['first_sentence']['sum_last_prob'], length)
            rep_sum_prob.add(rep_dict[target_repetition]['sum_last_prob'], length)
            improve_rate.add(rep_dict[target_repetition]['improve_count'], length)
            win_rate.add(rep_dict[target_repetition]['win_count'], length)
        rep_metric_dict[target_repetition] = {
            'First_AVG_Prob': first_sum_prob.mean(),
            'Repeat_AVG_Prob': rep_sum_prob.mean(),
            'Improve_Rate': improve_rate.mean(),
            'Win_Rate': win_rate.mean(),
            'Repetition' : target_repetition
        }

    return rep_metric_dict


if __name__ == '__main__':
    script_parser = argparse.ArgumentParser(
        description='Computes greedy completion, single-token prediction, and corresponding targets.')
    script_parser.add_argument('--pkl_dir', type=str, required=True)
    script_parser.add_argument('--save_dir', type=str, required=True)
    script_parser.add_argument('--draw_figures', type=int, default=0)

    high_level_args = script_parser.parse_args()

    data = main(file_dir=high_level_args.pkl_dir, draw_figures=high_level_args.draw_figures)
    data = pd.DataFrame(data=data)
    data.to_csv(os.path.join(high_level_args.save_dir, 'rep.csv'))
