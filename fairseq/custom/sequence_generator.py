# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from fairseq.utils import fill_with_neg_inf


def scaled_input(emb, batch_size, num_batch, baseline=None, start_i=None, end_i=None):
    # shape of emb: (1, num_head, seq_len, seq_len)
    if baseline is None:
        baseline = torch.zeros_like(emb)

    num_points = batch_size * num_batch
    scale = 1.0 / num_points
    if start_i is None:
        step = (emb - baseline) * scale
        res = torch.cat([torch.add(baseline, step*i) for i in range(num_points)], dim=0)
        return res, step[0]
    else:
        step = (emb - baseline) * scale
        start_emb = torch.add(baseline, step*start_i)
        end_emb = torch.add(baseline, step*end_i)
        step_new = (end_emb.unsqueeze(0) - start_emb.unsqueeze(0)) * scale
        res = torch.cat([torch.add(start_emb.unsqueeze(0), step_new*i) for i in range(num_points)], dim=0)
        return res, step_new[0]

class SequenceGenerator(object):
    def __init__(self, tgt_dict, temperature=1.):
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.temperature = temperature

    def generate_completion_greedy_training(self, model, prefix_tokens, completion_length):
        model.eval()
        # eval onyl have effects on the dropout and layernorm behavior but do not affect gradient behavior
        pred_toks = []
        context = prefix_tokens
        states = {}
        all_lprobs = []

        # First go over the context.
        for context_step in range(1, context.size(1)):
            _context = context[:, :context_step]
            _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)

        for tstep in range(completion_length):
            lprobs, attn_t = self._forward_one(model, context, incremental_states=states)
            pred_tok = lprobs.argmax(dim=1, keepdim=True)
            pred_toks.append(pred_tok)
            context = torch.cat((context, pred_tok), 1)
            all_lprobs.append(lprobs)

        pred_toks = torch.cat(pred_toks, 1)
        all_lprobs = torch.stack(all_lprobs, 1)
        return pred_toks, all_lprobs

    @torch.no_grad()
    def generate_completion(self, model, prefix_tokens, completion_length, topk, topp):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        context = prefix_tokens
        states = {}

        # First go over the context.
        for context_step in range(1, context.size(1)):
            _context = context[:, :context_step]
            _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)

        for tstep in range(completion_length):
            logits, attn_t = self._forward_one(model, context, incremental_states=states, return_logits=True)
            pred_tok = self._topk_decode(logits, topk, topp)
            pred_toks.append(pred_tok)
            context = torch.cat((context, pred_tok), 1)
        pred_toks = torch.cat(pred_toks, 1)
        return pred_toks


    def generate_next_token_prob(self, model, prefix_tokens, next_token_index):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        context = prefix_tokens

        with torch.no_grad():
            states = {}
            # First go over the context.
            for context_step in range(1, context.size(1)):
                _context = context[:, :context_step]
                _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)

            logits = self._forward_one_with_target_word(model, context, incremental_states=states, return_logits=True)

            logits[:, self.pad] = -1e10

            probs = torch.softmax(logits, -1)
            max_token = probs.argmax(dim=1)
            target_prob = probs[:, next_token_index]
            max_prob = probs[:, max_token][0]

            return target_prob, max_prob, max_token

    def generate_completion_with_logits_K(self, K, model, prefix_tokens, completion_length, topk, topp, batchsize=1, num_batch=1, require_attr=False):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        pred_toks_prob = []
        max_toks = []
        max_toks_prob = []
        context = prefix_tokens



        with torch.no_grad():
            states = {}
            # First go over the context.

            for context_step in range(1, context.size(1)):
                _context = context[:, :context_step]
                _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)


            for tstep in range(completion_length):
                logits, attn_t = self._forward_one(model, context, incremental_states=states, return_logits=True)
                pred_tok, pred_tok_prob, max_tok, max_tok_prob = self._topk_deterministic_decode(K, logits)
                pred_toks.append(pred_tok)
                pred_toks_prob.append(pred_tok_prob)
                max_toks.append(max_tok)
                max_toks_prob.append(max_tok_prob)
                context = torch.cat((context, pred_tok), 1)
            pred_toks = torch.cat(pred_toks, 1)
            pred_toks_prob = torch.cat(pred_toks_prob, 1)
            max_toks = torch.cat(max_toks, 1)
            max_toks_prob = torch.cat(max_toks_prob, 1)

            return pred_toks, pred_toks_prob, max_toks, max_toks_prob

    def generate_completion_with_logits_K_truncate(self, K, ratio, force_sample_top_truncate_K, force_sample_top_truncate_style, model, prefix_tokens, completion_length, topk, topp, batchsize=1, num_batch=1, require_attr=False):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        pred_toks_prob = []
        max_toks = []
        max_toks_prob = []
        context = prefix_tokens



        with torch.no_grad():
            states = {}
            # First go over the context.

            for context_step in range(1, context.size(1)):
                _context = context[:, :context_step]
                _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)


            for tstep in range(completion_length):
                logits, attn_t = self._forward_one(model, context, incremental_states=states, return_logits=True)
                if force_sample_top_truncate_style == 3:
                    probs = torch.softmax(logits, 1)
                    filtered_probs, bookkeep_idx = self._sample_topp(probs, sampling_topp=0.9)
                    filtered_probs = F.normalize(filtered_probs, p=1, dim=1)

                    # whether normalize ?
                    top1_indexs = bookkeep_idx[:, :1]
                    top1_indexs_prob = filtered_probs[:, :1]
                    topk_indexs = bookkeep_idx[:, 1:2]
                    topk_indexs_prob = filtered_probs[:, 1:2]
                    selected_index = torch.where(top1_indexs_prob > ratio, top1_indexs, topk_indexs)
                    selected_prob = torch.gather(probs, index=selected_index, dim=1)

                    pred_tok, pred_tok_prob, max_tok, max_tok_prob = selected_index, selected_prob , top1_indexs, top1_indexs_prob
                else:
                    pred_tok, pred_tok_prob, max_tok, max_tok_prob = self._topk_deterministic_decode_truncate(K, ratio, force_sample_top_truncate_K, force_sample_top_truncate_style, logits)
                pred_toks.append(pred_tok)
                pred_toks_prob.append(pred_tok_prob)
                max_toks.append(max_tok)
                max_toks_prob.append(max_tok_prob)
                context = torch.cat((context, pred_tok), 1)
            pred_toks = torch.cat(pred_toks, 1)
            pred_toks_prob = torch.cat(pred_toks_prob, 1)
            max_toks = torch.cat(max_toks, 1)
            max_toks_prob = torch.cat(max_toks_prob, 1)

            return pred_toks, pred_toks_prob, max_toks, max_toks_prob

    def generate_completion_with_logits(self, model, prefix_tokens, completion_length, topk, topp, batchsize=1, num_batch=1, require_attr=False):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        pred_toks_prob = []
        context = prefix_tokens

        if not require_attr:
            with torch.no_grad():
                states = {}
                # First go over the context.
                for context_step in range(1, context.size(1)):
                    _context = context[:, :context_step]
                    _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)

                for tstep in range(completion_length):
                    logits, attn_t = self._forward_one(model, context, incremental_states=states, return_logits=True)
                    pred_tok, pred_tok_prob = self._topk_decode(logits, topk, topp, return_prob=True)
                    pred_toks.append(pred_tok)
                    pred_toks_prob.append(pred_tok_prob)
                    context = torch.cat((context, pred_tok), 1)
                pred_toks = torch.cat(pred_toks, 1)
                pred_toks_prob = torch.cat(pred_toks_prob, 1)
                return pred_toks, pred_toks_prob
        else:
            assert topk == 1 and topp == 0

            starttime = datetime.now()
            all_attri_data = {}
            for tstep in range(completion_length):
                if tstep % 10 == 0:
                    print(f'Generate completition {tstep}-th step:', context)
                with torch.no_grad():
                    logits, attn_t = self._forward_one(model, context, return_logits=True, return_attn=True)
                    pred_tok, pred_tok_prob = self._topk_decode(logits, 1, 0, return_prob=True)
                    pred_toks.append(pred_tok)
                    pred_toks_prob.append(pred_tok_prob)
                    # context = torch.cat((context, pred_tok), 1)
                if context.size()[1] >= 310:
                    this_attr_batch = batchsize // 2
                    this_attr_num_batch = num_batch * 2
                elif context.size()[1] <= 155:
                    this_attr_batch = batchsize * 2
                    this_attr_num_batch = num_batch // 2
                else:
                    this_attr_batch = batchsize
                    this_attr_num_batch = num_batch

                num_layer = len(attn_t)
                pred_label = pred_tok.data

                res_attr = []
                for tar_layer in range(num_layer):
                    scale_att, step = scaled_input(attn_t[tar_layer].data, this_attr_batch, this_attr_num_batch)
                    scale_att.requires_grad_(True)

                    attr_all = None
                    for j_batch in range(this_attr_num_batch):
                        one_batch_att = scale_att[j_batch * this_attr_batch:(j_batch + 1) * this_attr_batch]
                        grad = self._forward_for_attr(model, context, tar_layer, one_batch_att, pred_label, return_logits=True)
                        grad = grad.sum(dim=0)
                        attr_all = grad if attr_all is None else torch.add(attr_all, grad)
                        # prob_all = tar_prob if prob_all is None else torch.cat([prob_all, tar_prob])
                    attr_all = attr_all * step
                    res_attr.append(attr_all.data.cpu().tolist())

                attri_data = {'attr_all': res_attr,
                              'context': context.data.cpu().tolist()[0],
                              'pre_tokens': pred_label.cpu().tolist()[0]}

                all_attri_data[tstep] = attri_data
                context = torch.cat((context, pred_tok), 1)
            endtime = datetime.now()
            print(f'Total time {(endtime - starttime).seconds}s for {completion_length} step completetion')

            pred_toks = torch.cat(pred_toks, 1)
            pred_toks_prob = torch.cat(pred_toks_prob, 1)
            return pred_toks, pred_toks_prob, all_attri_data

    def _forward_for_attr(self, model, tokens, tar_layer, tmp_score, pred_label, temperature=1.,  return_logits=False, **decoder_kwargs):
        decoder_out = list(model.decoder.attr_forward(tokens, tar_layer, tmp_score, encoder_out=None, return_attn=False, **decoder_kwargs))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)

        probs = model.get_normalized_probs(decoder_out, log_probs=False)
        probs = probs[:, -1, :]
        gradient = torch.autograd.grad(torch.unbind(probs[:, pred_label]), tmp_score)
        return gradient[0]

    def _sample_topp(self, probs, sampling_topp):
        """Sample among the smallest set of elements whose cumulative probability mass exceeds p.
        See `"The Curious Case of Neural Text Degeneration"
        (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.
        Args:
            probs: (bsz x input_beam_size x vocab_size)  IK: here we dont have beam ! so bsz x vocab_size
                the model's log-probabilities over the vocabulary at the current step
        Return: A tuple of (trimed_probs, truncated_indices) where:
            trimed_probs: (bsz x input_beam_size x ?)
                the model's probabilities over the elements selected to sample from. The
                width of the third dimension is determined by top-P.
            truncated_indices: (bsz x input_beam_size x ?)
                the indices of the chosen elements.
        """
        # sort the last dimension (vocab dimension) in descending order
        sorted_probs, sorted_indices = probs.sort(descending=True)

        # compute a mask to indicate the words to be included in the top-P set.
        cumsum_probs = sorted_probs.cumsum(dim=1)
        mask = cumsum_probs.lt(sampling_topp)

        # note that mask was computed by 'lt'. One more word needs to be included
        # so that the cumulative probability mass can exceed p.
        cumsum_mask = mask.cumsum(dim=1)
        last_included = cumsum_mask[:, -1:]
        last_included.clamp_(0, mask.size()[1] - 1)
        mask = mask.scatter_(1, last_included, 1)

        # truncate unnecessary dims.
        max_dim = last_included.max()
        truncated_mask = mask[:, :max_dim + 1]
        truncated_probs = sorted_probs[:, :max_dim + 1]
        truncated_indices = sorted_indices[:, :max_dim + 1]

        # trim the words that are not in top-P by setting their probabilities
        # to 0, so that they would not be sampled later.
        trim_mask = ~truncated_mask
        trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)
        return trimed_probs, truncated_indices

    def _topk_deterministic_decode(self, K, logits):
        logits[:, self.pad] = -math.inf  # as in fairseq code
        top1_indexs = torch.topk(logits, K)[1][:, 0].view(-1, 1)
        topk_indexs = torch.topk(logits, K)[1][:, K-1].view(-1, 1)
        probs = torch.softmax(logits, -1)
        return topk_indexs, torch.gather(probs, index=topk_indexs, dim=1) , top1_indexs, torch.gather(probs, index=top1_indexs, dim=1)

    def _topk_deterministic_decode_truncate(self, K, ratio, force_sample_top_truncate_K, force_sample_top_truncate_style, logits):
        logits[:, self.pad] = -math.inf  # as in fairseq code
        probs = torch.softmax(logits, -1)
        top1_indexs = torch.topk(logits, 3)[1][:, 0].view(-1, 1)
        top1_indexs_prob = torch.gather(probs, index=top1_indexs, dim=1)
        topk_indexs = torch.topk(logits, K)[1][:, K-1].view(-1, 1)
        topk_indexs_prob = torch.gather(probs, index=topk_indexs, dim=1)
        if force_sample_top_truncate_style == 1:
            if force_sample_top_truncate_K:
                selected_index = torch.where(topk_indexs_prob > ratio, topk_indexs, top1_indexs)
            else:
                selected_index = torch.where(top1_indexs_prob > ratio, top1_indexs, topk_indexs)
        elif force_sample_top_truncate_style == 2:
            top1_minus_topk = top1_indexs_prob - topk_indexs_prob
            selected_index = torch.where(top1_minus_topk > ratio, top1_indexs, topk_indexs)
        else:
            assert 1==0, 'not implemented error'
        selected_prob = torch.gather(probs, index=selected_index, dim=1)
        return selected_index, selected_prob , top1_indexs, top1_indexs_prob

    def _topk_decode(self, logits, topk, topp, return_prob=False):
        """WARNING!!! This can modify the `self.pad` position of `logits`."""

        if topk == 1 and topp == 0:  # greedy
            logits[:, self.pad] = -math.inf  # as in fairseq code
            pred_tok = logits.argmax(dim=1, keepdim=True)

        else:
            if topk > 1:
                logits[:, self.pad] = -1e10  # never select pad
                logits = top_k_logits(logits, topk)
                pred_tok = torch.softmax(logits, -1).multinomial(1)
            else:
                assert topp > 0.0
                filtered_probs, bookkeep_idx = self._sample_topp(torch.softmax(logits, 1), sampling_topp=topp)
                selected = filtered_probs.multinomial(1)
                pred_tok = torch.gather(bookkeep_idx, index=selected, dim=1)
        if return_prob:
            return pred_tok, torch.gather(torch.softmax(logits, -1), index=pred_tok, dim=1)
        return pred_tok

    def _forward_one(self, model, tokens, incremental_states=None, temperature=1., return_attn=False, return_logits=False, **decoder_kwargs):
        if incremental_states is not None:
            decoder_out = list(model.decoder(tokens, None, incremental_state=incremental_states, return_attn=return_attn, **decoder_kwargs))
        else:
            decoder_out = list(model.decoder(tokens, None, return_attn=return_attn, **decoder_kwargs))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn['attn']
        # if attn is not None:
        #     if type(attn) is dict:
        #         attn = attn['attn']
        #     attn = attn[:, :, -1, :]  # B x L x t
        if return_logits:
            logits_t = decoder_out[0][:, -1, :]
            return logits_t, attn
        log_probs = model.get_normalized_probs(decoder_out, log_probs=True)
        log_probs = log_probs[:, -1, :]
        return log_probs, attn

    def _forward_one_with_target_word(self, model, tokens, incremental_states=None, temperature=1., return_attn=False, return_logits=False, **decoder_kwargs):
        if incremental_states is not None:
            decoder_out = list(model.decoder(tokens, None, incremental_state=incremental_states, return_attn=return_attn, **decoder_kwargs))
        else:
            decoder_out = list(model.decoder(tokens, None, return_attn=return_attn, **decoder_kwargs))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)

        if return_logits:
            logits_t = decoder_out[0][:, -1, :]
            return logits_t

    # **For intra_sentence obtain the attention mask**
    def _gen_intra_sentence_mask(self, tokens, vocab, last=True):
        batch_size = tokens.size()[0]
        o_tokens = tokens.cpu()

        masks = []
        for idx in range(batch_size):
            mask = self._intra_sentence_mask(o_tokens[idx], vocab)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        if last:
            masks = masks[:, -1:]

        masks = fill_with_neg_inf(torch.zeros(masks.size())).masked_fill_(masks, 0)
        masks = masks.to(device=tokens.device)

        return masks

    def _intra_sentence_mask(self, source, vocab):
        source_list = source.tolist()
        source_len = len(source_list)
        text = [vocab[s] for s in source_list]
        sentence_start_end_indexs = np.zeros((len(source_list), 2), dtype=int)
        sentence_end_indexs = []

        sentence_end_index = vocab.index('.')
        for idx, token in enumerate(source_list):
            if token == sentence_end_index:
                sentence_end_indexs.append(idx)

        cls_token_masks = []
        last_pointer, current_pointer = None, 0
        for idx, token in enumerate(source_list):
            if (current_pointer < len(sentence_end_indexs)) and (idx > sentence_end_indexs[current_pointer]):
                last_pointer = current_pointer
                current_pointer += 1

            if last_pointer is not None:
                sentence_start_end_indexs[idx][0] = sentence_end_indexs[last_pointer]

            # if current_pointer < len(sentence_end_indexs):
            #     sentence_start_end_indexs[idx][1] = sentence_end_indexs[current_pointer]
            # else:
            sentence_start_end_indexs[idx][1] = idx

            cls_token_mask = np.zeros((source_len), dtype=bool)
            cls_token_mask[sentence_end_indexs[:current_pointer]] = True
            cls_token_masks.append(cls_token_mask)

        cls_token_masks = torch.tensor(np.array(cls_token_masks), dtype=torch.bool, device=source.device)


        pair_index = torch.LongTensor(sentence_start_end_indexs, device=source.device)
        cols = torch.LongTensor(range(source_len)).repeat(source_len, 1)
        beg = pair_index[:, 0].unsqueeze(1).repeat(1, source_len)
        end = pair_index[:, 1].unsqueeze(1).repeat(1, source_len)
        intra_sen_mask = cols.ge(beg) & cols.le(end)

        mask = cls_token_masks | intra_sen_mask

        return mask

    def _forward_intra_one(self, model, tokens, vocab, incremental_states=None, temperature=1., return_attn=False, return_logits=False, **decoder_kwargs):
        if incremental_states is not None:
            src_attn_mask = self._gen_intra_sentence_mask(tokens, vocab, last=True)
            decoder_out = list(model.decoder(tokens, None, incremental_state=incremental_states, src_attn_mask=src_attn_mask, return_attn=return_attn, **decoder_kwargs))
        else:
            src_attn_mask = self._gen_intra_sentence_mask(tokens, vocab, last=False)
            decoder_out = list(model.decoder(tokens, None, src_attn_mask=src_attn_mask, return_attn=return_attn, **decoder_kwargs))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn['attn']
        # if attn is not None:
        #     if type(attn) is dict:
        #         attn = attn['attn']
        #     attn = attn[:, :, -1, :]  # B x L x t
        if return_logits:
            logits_t = decoder_out[0][:, -1, :]
            return logits_t, attn
        log_probs = model.get_normalized_probs(decoder_out, log_probs=True)
        log_probs = log_probs[:, -1, :]
        return log_probs, attn

    @torch.no_grad()
    def generate_intra_completion(self, model, vocab, prefix_tokens, completion_length, topk, topp):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        context = prefix_tokens
        states = {}
        # First go over the context.
        for context_step in range(1, context.size(1)):
            _context = context[:, :context_step]
            _ = self._forward_intra_one(model, _context, vocab, incremental_states=states, return_logits=True)

        for tstep in range(completion_length):
            logits, attn_t = self._forward_intra_one(model, context, vocab, incremental_states=states, return_logits=True)
            pred_tok = self._topk_decode(logits, topk, topp)
            pred_toks.append(pred_tok)
            context = torch.cat((context, pred_tok), 1)
        pred_toks = torch.cat(pred_toks, 1)
        return pred_toks

    def generate_intra_completion_with_logits(self, model, prefix_tokens, completion_length, topk, topp, vocab, batchsize=1, num_batch=1, require_attr=False):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        pred_toks_prob = []
        context = prefix_tokens

        if not require_attr:
            with torch.no_grad():
                states = {}
                # First go over the context.
                for context_step in range(1, context.size(1)):
                    _context = context[:, :context_step]
                    _ = self._forward_intra_one(model, _context, vocab, incremental_states=states, return_logits=True)

                for tstep in range(completion_length):
                    logits, _ = self._forward_intra_one(model, context, vocab, incremental_states=states, return_logits=True)
                    pred_tok, pred_tok_prob = self._topk_decode(logits, topk, topp, return_prob=True)
                    pred_toks.append(pred_tok)
                    pred_toks_prob.append(pred_tok_prob)
                    context = torch.cat((context, pred_tok), 1)
                pred_toks = torch.cat(pred_toks, 1)
                pred_toks_prob = torch.cat(pred_toks_prob, 1)
                return pred_toks, pred_toks_prob
        else:
            assert topk == 1 and topp == 0

            starttime = datetime.now()
            all_attri_data = {}
            # for debug
            for tstep in range(completion_length):
                if tstep % 10 == 0:
                    print(f'Generate completition {tstep}-th step:', context)

                with torch.no_grad():
                    logits, attn_t = self._forward_intra_one(model, context, vocab, return_logits=True, return_attn=True)
                    pred_tok, pred_tok_prob = self._topk_decode(logits, 1, 0, return_prob=True)
                    pred_toks.append(pred_tok)
                    pred_toks_prob.append(pred_tok_prob)
                    # context = torch.cat((context, pred_tok), 1)
                if context.size()[1] >= 310:
                    this_attr_batch = batchsize // 2
                    this_attr_num_batch = num_batch * 2
                elif context.size()[1] <= 155:
                    this_attr_batch = batchsize * 2
                    this_attr_num_batch = num_batch // 2
                else:
                    this_attr_batch = batchsize
                    this_attr_num_batch = num_batch

                num_layer = len(attn_t)
                pred_label = pred_tok.data

                res_attr = []
                for tar_layer in range(num_layer):
                    scale_att, step = scaled_input(attn_t[tar_layer].data, this_attr_batch, this_attr_num_batch)
                    scale_att.requires_grad_(True)

                    attr_all = None
                    for j_batch in range(this_attr_num_batch):
                        one_batch_att = scale_att[j_batch * this_attr_batch:(j_batch + 1) * this_attr_batch]
                        grad = self._forward_for_intra_attr(model, context, vocab, tar_layer, one_batch_att, pred_label, return_logits=True)
                        grad = grad.sum(dim=0)
                        attr_all = grad if attr_all is None else torch.add(attr_all, grad)
                        # prob_all = tar_prob if prob_all is None else torch.cat([prob_all, tar_prob])
                    attr_all = attr_all * step
                    res_attr.append(attr_all.data.cpu().tolist())

                attri_data = {'attr_all': res_attr,
                              'context': context.data.cpu().tolist()[0],
                              'pre_tokens': pred_label.cpu().tolist()[0]}

                all_attri_data[tstep] = attri_data
                context = torch.cat((context, pred_tok), 1)
            endtime = datetime.now()
            print(f'Total time {(endtime - starttime).seconds}s for {completion_length} step completetion')

            pred_toks = torch.cat(pred_toks, 1)
            pred_toks_prob = torch.cat(pred_toks_prob, 1)
            return pred_toks, pred_toks_prob, all_attri_data

    def _forward_for_intra_attr(self, model, tokens, vocab, tar_layer, tmp_score, pred_label, temperature=1.,  return_logits=False, **decoder_kwargs):
        src_attn_mask = self._gen_intra_sentence_mask(tokens, vocab, last=False)
        decoder_out = list(model.decoder.attr_forward(tokens, tar_layer, tmp_score, src_attn_mask=src_attn_mask, encoder_out=None, return_attn=False, **decoder_kwargs))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)

        probs = model.get_normalized_probs(decoder_out, log_probs=False)
        probs = probs[:, -1, :]
        gradient = torch.autograd.grad(torch.unbind(probs[:, pred_label]), tmp_score)
        return gradient[0]

    def generate_intra_completion_with_logits_K(self, K, model, prefix_tokens, vocab, completion_length, topk, topp, batchsize=1, num_batch=1, require_attr=False):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        pred_toks_prob = []
        max_toks = []
        max_toks_prob = []
        context = prefix_tokens

        with torch.no_grad():
            states = {}
            # First go over the context.

            for context_step in range(1, context.size(1)):
                _context = context[:, :context_step]
                _ = self._forward_intra_one(model, _context, vocab, incremental_states=states, return_logits=True)


            for tstep in range(completion_length):
                logits, attn_t = self._forward_intra_one(model, context, vocab, incremental_states=states, return_logits=True)
                pred_tok, pred_tok_prob, max_tok, max_tok_prob = self._topk_deterministic_decode(K, logits)
                pred_toks.append(pred_tok)
                pred_toks_prob.append(pred_tok_prob)
                max_toks.append(max_tok)
                max_toks_prob.append(max_tok_prob)
                context = torch.cat((context, pred_tok), 1)
            pred_toks = torch.cat(pred_toks, 1)
            pred_toks_prob = torch.cat(pred_toks_prob, 1)
            max_toks = torch.cat(max_toks, 1)
            max_toks_prob = torch.cat(max_toks_prob, 1)

            return pred_toks, pred_toks_prob, max_toks, max_toks_prob

    # **END - For intra_sentence obtain the attention mask**
    # **START - For greedy block algorithm**
    def _block_logits(self, context, logits, n_gram_block):
        if context.size()[1] <=n_gram_block:
            return logits
        n = n_gram_block - 1
        for i, x in enumerate(context):
            masked_indexs = set()
            xl = x.tolist()
            to_match = tuple(xl[-n: ])
            for j in range(len(x) - n):
                if tuple(xl[j:j + n]) == to_match:
                    masked_indexs.add(xl[j+n])
            masked_indexs = list(masked_indexs)
            if len(masked_indexs) != 0:
                logits[i, masked_indexs] = -math.inf
        return logits

    @torch.no_grad()
    def generate_completion_block(self, model, n_gram_block, prefix_tokens, completion_length, topk, topp):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        context = prefix_tokens
        states = {}

        # First go over the context.
        for context_step in range(1, context.size(1)):
            _context = context[:, :context_step]
            _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)

        for tstep in range(completion_length):
            logits, _ = self._forward_one(model, context, incremental_states=states, return_logits=True)
            logits = self._block_logits(context, logits, n_gram_block)
            pred_tok = self._topk_decode(logits, topk, topp)
            pred_toks.append(pred_tok)
            context = torch.cat((context, pred_tok), 1)
        pred_toks = torch.cat(pred_toks, 1)
        return pred_toks

    # **END - For greedy block algorithm**

    # **START - For two step gen**
    @torch.no_grad()
    def generate_tsg_completion(self, model, prefix_tokens, completion_length, two_step_gen):
        """two-step generation"""
        model.eval()
        pred_toks = []
        context = prefix_tokens
        states = {}

        num_of_expand = 5

        # First go over the context.
        for context_step in range(1, context.size(1)):
            _context = context[:, :context_step]
            _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)

        p_hat = None
        for tstep in range(completion_length):
            # states['MultiheadAttention.1.attn_state']['prev_key'].size()
            if tstep == 0:
                logits, attn_t = self._forward_one(model, context, incremental_states=states, return_logits=True)
                probs = torch.softmax(logits, 1)
                filtered_probs, bookkeep_idx = self._sample_topp(probs, sampling_topp=0.9)
                all_next_probs = []
                for i in range(1, num_of_expand):
                    tmp_pred_tok = bookkeep_idx[:, i:i+1]
                    tmp_context = torch.cat((context, tmp_pred_tok), 1)
                    tmp_logits, _ = self._forward_one(model, tmp_context, incremental_states=states, return_logits=True, incremental_update=False)
                    tmp_probs = torch.softmax(tmp_logits, 1)
                    all_next_probs.append(tmp_probs)
                all_next_probs = torch.stack(all_next_probs, dim=0)
                all_next_probs = all_next_probs.transpose(1, 0) # B * num_of_expand * (T - 1)
                current_weight_probs = filtered_probs[:, 1:num_of_expand] # B * (T - 1)
                current_weight_probs = current_weight_probs.unsqueeze(1)
                p_hat = torch.matmul(current_weight_probs, all_next_probs)
                p_hat = p_hat.squeeze(1)

                pred_toks.append(bookkeep_idx[:, :1])
                context = torch.cat((context, bookkeep_idx[:, :1]), 1)
            else:
                logits, attn_t = self._forward_one(model, context, incremental_states=states, return_logits=True)
                m_probs = torch.softmax(logits, 1)

                probs = two_step_gen * p_hat + (1 - two_step_gen) * m_probs
                probs = F.normalize(probs, dim=1, p=1)

                filtered_probs, bookkeep_idx = self._sample_topp(probs, sampling_topp=0.9)
                all_next_probs = []
                for i in range(1, num_of_expand):
                    tmp_pred_tok = bookkeep_idx[:, i:i + 1]
                    tmp_context = torch.cat((context, tmp_pred_tok), 1)
                    tmp_logits, _ = self._forward_one(model, tmp_context, incremental_states=states, return_logits=True,
                                                      incremental_update=False)
                    tmp_probs = torch.softmax(tmp_logits, 1)
                    all_next_probs.append(tmp_probs)
                all_next_probs = torch.stack(all_next_probs, dim=0)
                all_next_probs = all_next_probs.transpose(1, 0)  # B * num_of_expand * (T-1)
                current_weight_probs = filtered_probs[:, 1:num_of_expand]  # B * (T-1)
                current_weight_probs = current_weight_probs.unsqueeze(1)
                p_hat = torch.matmul(current_weight_probs, all_next_probs)
                p_hat = p_hat.squeeze(1)

                pred_toks.append(bookkeep_idx[:, :1])
                context = torch.cat((context, bookkeep_idx[:, :1]), 1)


        pred_toks = torch.cat(pred_toks, 1)
        return pred_toks

    def generate_tsg_completion_with_logits_K(self, two_step_gen, model, prefix_tokens, completion_length, topk, topp, batchsize=1, num_batch=1, require_attr=False):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        pred_toks_prob = []
        max_toks = []
        max_toks_prob = []
        context = prefix_tokens

        num_of_expand = 5
        with torch.no_grad():
            states = {}
            # First go over the context.

            for context_step in range(1, context.size(1)):
                _context = context[:, :context_step]
                _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)

            p_hat = None
            for tstep in range(completion_length):
                if tstep == 0:
                    logits, attn_t = self._forward_one(model, context, incremental_states=states, return_logits=True)
                    probs = torch.softmax(logits, 1)
                    filtered_probs, bookkeep_idx = self._sample_topp(probs, sampling_topp=1.0)
                    all_next_probs = []
                    for i in range(1, min(num_of_expand, filtered_probs.size()[1])):
                        tmp_pred_tok = bookkeep_idx[:, i:i + 1]
                        tmp_context = torch.cat((context, tmp_pred_tok), 1)
                        tmp_logits, _ = self._forward_one(model, tmp_context, incremental_states=states,
                                                          return_logits=True, incremental_update=False)
                        tmp_probs = torch.softmax(tmp_logits, 1)
                        all_next_probs.append(tmp_probs)
                    all_next_probs = torch.stack(all_next_probs, dim=0)
                    all_next_probs = all_next_probs.transpose(1, 0)  # B * num_of_expand * (T - 1)
                    current_weight_probs = filtered_probs[:, 1:num_of_expand]  # B * (T - 1)
                    current_weight_probs = current_weight_probs.unsqueeze(1)
                    p_hat = torch.matmul(current_weight_probs, all_next_probs)
                    p_hat = p_hat.squeeze(1)

                    pred_toks.append(bookkeep_idx[:, :1])
                    pred_toks_prob.append(filtered_probs[:, :1])
                    max_toks.append(bookkeep_idx[:, :1])
                    max_toks_prob.append(filtered_probs[:, :1])

                    context = torch.cat((context, bookkeep_idx[:, :1]), 1)
                else:
                    logits, attn_t = self._forward_one(model, context, incremental_states=states, return_logits=True)
                    m_probs = torch.softmax(logits, 1)

                    probs = two_step_gen * p_hat + (1 - two_step_gen) * m_probs
                    probs = F.normalize(probs, dim=1, p=1)

                    filtered_probs, bookkeep_idx = self._sample_topp(probs, sampling_topp=1.0)
                    if filtered_probs.size()[1] != 1:
                        all_next_probs = []
                        for i in range(1, min(num_of_expand, filtered_probs.size()[1])):
                            tmp_pred_tok = bookkeep_idx[:, i:i + 1]
                            tmp_context = torch.cat((context, tmp_pred_tok), 1)
                            tmp_logits, _ = self._forward_one(model, tmp_context, incremental_states=states,
                                                              return_logits=True,
                                                              incremental_update=False)
                            tmp_probs = torch.softmax(tmp_logits, 1)
                            all_next_probs.append(tmp_probs)
                        all_next_probs = torch.stack(all_next_probs, dim=0)

                        all_next_probs = all_next_probs.transpose(1, 0)  # B * num_of_expand * (T-1)
                        current_weight_probs = filtered_probs[:, 1:num_of_expand]  # B * (T-1)
                        current_weight_probs = current_weight_probs.unsqueeze(1)
                        p_hat = torch.matmul(current_weight_probs, all_next_probs)
                        p_hat = p_hat.squeeze(1)
                    else:
                        p_hat = 0

                    pred_toks.append(bookkeep_idx[:, :1])
                    pred_toks_prob.append(filtered_probs[:, :1])

                    m_filtered_probs, m_bookkeep_idx = self._sample_topp(m_probs, sampling_topp=1.0)

                    max_toks.append(m_bookkeep_idx[:, :1])
                    max_toks_prob.append(m_filtered_probs[:, :1])
                    context = torch.cat((context, bookkeep_idx[:, :1]), 1)


            pred_toks = torch.cat(pred_toks, 1)
            pred_toks_prob = torch.cat(pred_toks_prob, 1)
            max_toks = torch.cat(max_toks, 1)
            max_toks_prob = torch.cat(max_toks_prob, 1)

            return pred_toks, pred_toks_prob, max_toks, max_toks_prob

    # **End - For two step gen**
    # ** Start - See more candidate for more insights **

    def _topk_deterministic_decode_return_more_prob(self, K, show_K, logits):
        logits[:, self.pad] = -math.inf  # as in fairseq code
        probs = torch.softmax(logits, -1)
        show_probs, show_indexs = torch.topk(probs, show_K)
        topk_indexs = torch.topk(probs, K)[1][:, K-1].view(-1, 1)
        return topk_indexs, torch.gather(probs, index=topk_indexs, dim=1) , show_indexs, show_probs

    def get_token_similarity(self, tokens, model):
        # tokens: B * T
        tok_embeddings = model.decoder.embed_tokens(tokens)
        similarity_matrix = F.cosine_similarity(tok_embeddings[..., None, :, :], tok_embeddings[..., :, None, :], dim=-1)
        ad_sim = torch.diagonal(similarity_matrix, offset=1, dim1=1, dim2=2)
        return ad_sim



    def generate_completion_with_more_logits(self, K, show_K, model, prefix_tokens, completion_length, topk, topp, batchsize=1, num_batch=1, require_attr=False):
        """topk: <1 sampling, 1 greedy, >1 top-k sampling."""
        model.eval()
        pred_toks = []
        pred_toks_prob = []
        max_toks = []
        max_toks_prob = []
        ad_sim_all = []
        context = prefix_tokens

        with torch.no_grad():
            states = {}
            # First go over the context.

            for context_step in range(1, context.size(1)):
                _context = context[:, :context_step]
                _ = self._forward_one(model, _context, incremental_states=states, return_logits=True)


            for tstep in range(completion_length):
                logits, attn_t = self._forward_one(model, context, incremental_states=states, return_logits=True)
                pred_tok, pred_tok_prob, show_toks, show_probs = self._topk_deterministic_decode_return_more_prob(K, show_K, logits)
                ad_sim = self.get_token_similarity(show_toks, model)
                pred_toks.append(pred_tok)
                pred_toks_prob.append(pred_tok_prob)
                max_toks.append(show_toks)
                max_toks_prob.append(show_probs)
                ad_sim_all.append(ad_sim)
                context = torch.cat((context, pred_tok), 1)
            pred_toks = torch.cat(pred_toks, 1)
            pred_toks_prob = torch.cat(pred_toks_prob, 1)
            max_toks = torch.stack(max_toks, 1)
            max_toks_prob = torch.stack(max_toks_prob, 1)
            ad_sim_all = torch.stack(ad_sim_all, 1)

            return pred_toks, pred_toks_prob, max_toks, max_toks_prob, ad_sim_all




def top_k_logits(logits, k):
    """Masks everything but the k top entries as -infinity (1e10).
    From: https://github.com/huggingface/pytorch-pretrained-BERT"""
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e5, logits)