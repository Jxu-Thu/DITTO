# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
import torch.nn.functional as F


from collections import defaultdict
from fairseq import utils
import random

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.custom.evaluate_utils import batch_input_sequence_by_prefix_length
from fairseq.custom.metrics import ngram_metrics, TrainingMetrics


@register_criterion('repetetion_accum_penalty')
class RepetetionPenaltyAccumCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        # self.sequence_ngram_n = args.sequence_ngram_n
        # self.sequence_prefix_length = args.sequence_prefix_length
        # self.sequence_completion_length = args.sequence_completion_length
        # self.sequence_candidate_type = args.sequence_candidate_type
        # self.mask_p = args.mask_p
        self.rep_reduce_gamma = args.rep_reduce_gamma # repetetion prob discount
        self.end_sentence_decoded = self.task.target_dictionary.index('.')
        self.loss_type = args.loss_type

    def re_orgnize_sentence(self, sample):
        # First ,random pick a sentence in a batch
        # Then, get sentence length, L, repeat N times to get maximun len, calculate remains K
        # Last, return the batch of samples

        max_tokens = sample['net_input']['src_tokens'].size(1)
        P = []
        L = []
        N = []
        K = []
        ALL_TOKENS = []
        TARGET_TOKENS = []
        for i, x in enumerate(sample['net_input']['src_tokens']):
            xl = x.tolist()
            sentence_end_indexs = []
            for idx, token in enumerate(xl):
                if token == self.end_sentence_decoded:
                    sentence_end_indexs.append(idx)
            try:
                sen_idx = random.randint(1, len(sentence_end_indexs) - 2)
                last_sen_start = sentence_end_indexs[sen_idx - 1] + 1
                sen_start = sentence_end_indexs[sen_idx]
                sen_end = sentence_end_indexs[sen_idx + 1]
            except:
                return None, None, None, None, None

            prefix = x[last_sen_start: sen_start]
            prefix_len = sen_start - last_sen_start
            left_tokens = max_tokens - prefix_len

            x_senetence = x[sen_start: sen_end].view(1, -1)
            sen_len = sen_end - sen_start
            n, k = left_tokens//sen_len, left_tokens%sen_len
            x_senetence = x_senetence.repeat(n + 1, 1).view(-1)
            new_sentence = torch.cat([prefix, x_senetence], dim=0)
            input_senetence = new_sentence[:max_tokens]
            target_sentence = new_sentence[1:max_tokens+1]
            assert target_sentence.size()[0] == max_tokens
            P.append(sen_start - last_sen_start)
            N.append(n)
            K.append(k)
            L.append(sen_len)
            ALL_TOKENS.append(input_senetence)
            TARGET_TOKENS.append(target_sentence)
        ALL_TOKENS = torch.stack(ALL_TOKENS, dim=0)
        TARGET_TOKENS = torch.stack(TARGET_TOKENS, dim=0)

        sample['net_input']['src_tokens'] = ALL_TOKENS
        sample['target'] = TARGET_TOKENS
        return sample, P, L, N, K


    def forward(self, model, sample, reduce=True, compute_custom_metrics=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample, P, L, N, K = self.re_orgnize_sentence(sample)
        if sample is None:
            return None, None, None, True

        net_output = model(**sample['net_input'])


        # logits = net_output[0].view(-1, net_output[0].size(-1))
        # target = model.get_targets(sample, net_output)
        # target = target.view(-1)
        loss, _, sample_size = self.compute_loss(model, net_output, sample, P, L, N, K, reduce=reduce)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample_size,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        # if compute_custom_metrics:
        #     custom_output = TrainingMetrics.ranking_metrics(logits, true_token_logits, sample, ntokens, target)
        #     for k, v in custom_output.items():
        #         logging_output[k] = v
        return loss, sample_size, logging_output, False


    def compute_loss(self, model, net_output, sample, P, L, N, K, reduce=True):
        probs = model.get_normalized_probs(net_output, log_probs=False)
        B, T, vocab_size = probs.size()
        probs = probs.view(B * T, vocab_size)
        target = model.get_targets(sample, net_output).view(-1, 1)
        target_probs = probs.gather(1, target).view(B, T)
        gt_probs, mask, valid_tokens = self.obtain_rep_baseline_prob(model.get_targets(sample, net_output),
                                                                     target_probs.detach(),
                                                                     P, L, N, K)
        if self.loss_type == 'mse':
            loss = F.mse_loss(target_probs, gt_probs*self.rep_reduce_gamma, reduction='none')
            loss = loss.sum()
            loss = loss * 3 # loss scale is smaller than nll
            return loss, loss, valid_tokens
        elif self.loss_type == 'nl':
            one_minus_probs = torch.clamp((1.0 - torch.abs((target_probs - gt_probs*self.rep_reduce_gamma))), min=1e-20)
            loss = -torch.log(one_minus_probs) * mask
            loss = loss.sum()
            return loss, loss, valid_tokens
        elif self.loss_type == 'nl_clip':
            one_minus_probs = torch.clamp((1.0 - torch.clamp(target_probs - gt_probs*self.rep_reduce_gamma, min=0.0)), min=1e-20)
            loss = -torch.log(one_minus_probs) * mask
            loss = loss.sum()
            return loss, loss, valid_tokens
        else:
            assert 1==0, 'not implemented error'





        # ---------------------
        # lprobs = model.get_normalized_probs(net_output, log_probs=True)
        #
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        # target = model.get_targets(sample, net_output).view(-1)
        # loss = F.nll_loss(
        #     lprobs,
        #     target,
        #     ignore_index=self.padding_idx,
        #     reduction='sum' if reduce else 'none',
        # )
        # return loss, loss

    def obtain_rep_baseline_prob(self, target, target_probs, P, L, N, K):
        max_tokens = target_probs.size(1)
        gt_probs = []
        mask = []
        valid_tokens = 0

        for i, x in enumerate(target_probs):
            prefix_len = P[i]
            rep_len = L[i]
            repeat_times = N[i]
            remain_tokens = K[i]
            start_sen_prob = x[prefix_len: ]
            to_pelize_tokens = rep_len * (repeat_times - 1) + remain_tokens
            new_cp_probs = torch.zeros_like(x)
            new_cp_probs[prefix_len+rep_len:] = start_sen_prob[:to_pelize_tokens]
            gt_probs.append(new_cp_probs)
            this_mask = torch.zeros_like(x, dtype=torch.bool)
            this_mask[prefix_len+rep_len:] = True
            mask.append(this_mask)
            valid_tokens += to_pelize_tokens
        gt_probs = torch.stack(gt_probs, dim=0)
        mask = torch.stack(mask, dim=0)
        return gt_probs, mask, valid_tokens




    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        from fairseq.custom.metrics import TrainingMetrics
        custom_output = TrainingMetrics.aggregate_and_normalize(logging_outputs)
        for k, v in custom_output.items():
            agg_output[k] = v

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2) if ntokens > 0 else 0.
        return agg_output



