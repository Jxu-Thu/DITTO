# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter
from fairseq.custom.metrics import Metrics, TrainingMetrics
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from fairseq.utils import fill_with_neg_inf

import math

def load(args, task=None, itr=None, generator=None, log=False, random_net_init=False, re_init_weight=None):
    """Returns task, model, generator, and dataset iterator for the given `args`."""
    assert args.path is not None, '--path required for generation!'
    import random
    random.seed(42)
    torch.manual_seed(42)
    utils.import_user_module(args)
    if log:
        print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    if task is None:
        task = tasks.setup_task(args)
        task.load_dataset(args.gen_subset)

    # Load ensemble
    if log:
        print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
        random_net_init=random_net_init,
        re_init_weight=re_init_weight,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
    model = models[0]

    if itr is None:
        # Load dataset (possibly sharded)
        itr = task.get_batch_iterator(
            dataset=task.dataset(args.gen_subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=args.tokens_per_sample,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

    # Get model step
    step = torch.load(args.path)['optimizer_history'][-1]['num_updates']

    if generator is None:
        # Initialize generator
        generator = task.build_generator(args)
    return task, model, generator, itr, step


def generate_completions(model, generator, fairseq_generator, itr, eval_prefix_length, eval_completion_length, topk, topp, num_samples, beam_size, n_gram_block=0, include_prefix=True, two_step_gen=False):
    completions = []
    gt_completion = []
    completion_metrics = Metrics()
    actual_metrics = Metrics()
    for n, sample in enumerate(tqdm(itr)):
        input_sequence = sample['net_input']['src_tokens']
        prefix_batch, gt_com = batch_input_sequence_by_prefix_length_and_align_target(input_sequence, eval_prefix_length, eval_completion_length)
        prefix_batch = prefix_batch.cuda()
        if input_sequence.size(1) < eval_prefix_length:
            continue
        if beam_size > 1:
            assert topk == 1, 'with greedy topk must be 1'
            assert topp == 0.0, 'with greedy topp must be 0'
            sample['net_input']['src_tokens'] = prefix_batch
            res = fairseq_generator.generate([model], sample, prefix_batch, bos_token=0)  # prefix is there in preds!
            pred_completion = [res[i][0]['tokens'][eval_prefix_length:-1].cpu().tolist() for i in range(len(res))]
        elif beam_size == 1:
            if n_gram_block > 0:
                assert n_gram_block > 1, 'at least >= 2'
                pred_completion = generator.generate_completion_block(model, n_gram_block, prefix_batch, eval_completion_length, topk, topp)
                pred_completion = pred_completion.cpu().tolist()
                # write logic here
            elif two_step_gen > 0:
                pred_completion = generator.generate_tsg_completion(model, prefix_batch, eval_completion_length, two_step_gen)
                pred_completion = pred_completion.cpu().tolist()

            else:
                pred_completion = generator.generate_completion(model, prefix_batch, eval_completion_length, topk, topp)
                pred_completion = pred_completion.cpu().tolist()
        # Calcuate unique n-grams / total grams metrics
        completion_metrics.update(pred_completion)
        actual_metrics.update(input_sequence)

        if include_prefix:
            prefix_batch = prefix_batch.cpu().tolist()
            pred_completion = [prefix + completion for
                               prefix, completion in zip(prefix_batch, pred_completion)]
            gt_com = [prefix + completion for
                               prefix, completion in zip(prefix_batch, gt_com)]

        completions.extend(pred_completion)
        gt_completion.extend(gt_com)

        if n == num_samples:
            break

    completion_metrics = completion_metrics.report('generated')
    actual_metrics = actual_metrics.report('actual')
    return completions, gt_completion, completion_metrics, actual_metrics

def single_completions(sentence, model, generator, fairseq_generator, eval_prefix_length, eval_completion_length, topk, topp, beam_size, batchsize=1, num_batch=1, require_attr=False):
    completions = []
    completions_probs = []
    completion_metrics = Metrics()
    sample = {
        'id': torch.tensor([1]),
        'nsentences': 1,
        'ntokens': len(sentence),
        'net_input':{
            'src_tokens': torch.tensor(sentence).view(1, -1),
            'src_lengths': torch.tensor([len(sentence)])
        }
    }
    input_sequence = sample['net_input']['src_tokens']
    prefix_batch = input_sequence.cuda()

    if beam_size > 1:
        assert topk == 1, 'with greedy topk must be 1'
        assert topp == 0.0, 'with greedy topp must be 0'
        sample['net_input']['src_tokens'] = prefix_batch
        res = fairseq_generator.generate([model], sample, prefix_batch, bos_token=0)  # prefix is there in preds!
        pred_completion = [res[i][0]['tokens'][eval_prefix_length:-1].cpu().tolist() for i in range(len(res))]
        pred_completion_probs = [res[i][0]['positional_scores'][eval_prefix_length:-1].exp().cpu().tolist() for i in range(len(res))]

        # Calcuate unique n-grams / total grams metrics
        completion_metrics.update(pred_completion)

        completions.extend(pred_completion)
        completions_probs.extend(pred_completion_probs)

        completion_metrics = completion_metrics.report('generated')
        return completions, completion_metrics, completions_probs, None

    elif beam_size == 1:

        return_results = generator.generate_completion_with_logits(model, prefix_batch, eval_completion_length, topk, topp,
                                                                                           batchsize=batchsize, num_batch=num_batch, require_attr=require_attr)
        all_attri_data = None
        if require_attr:
            pred_completion, pred_completion_probs, all_attri_data = return_results
        else:
            pred_completion, pred_completion_probs = return_results
        pred_completion = pred_completion.cpu().tolist()
        pred_completion_probs = pred_completion_probs.cpu().tolist()

        # Calcuate unique n-grams / total grams metrics
        completion_metrics.update(pred_completion)

        completions.extend(pred_completion)
        completions_probs.extend(pred_completion_probs)

        completion_metrics = completion_metrics.report('generated')
        return completions, completion_metrics, completions_probs, all_attri_data

def single_completions_with_k_candidate(sentence, model, generator, K, eval_prefix_length, eval_completion_length, topk, topp, beam_size, batchsize=1, num_batch=1, require_attr=False, two_step_gen=0):
    completions = []
    completions_probs = []
    completion_metrics = Metrics()
    sample = {
        'id': torch.tensor([1]),
        'nsentences': 1,
        'ntokens': len(sentence),
        'net_input':{
            'src_tokens': torch.tensor(sentence).view(1, -1),
            'src_lengths': torch.tensor([len(sentence)])
        }
    }
    input_sequence = sample['net_input']['src_tokens']
    prefix_batch = input_sequence.cuda()

    if two_step_gen > 0:
        return_results = generator.generate_tsg_completion_with_logits_K(two_step_gen, model, prefix_batch, eval_completion_length,
                                                                     topk, topp,
                                                                     batchsize=batchsize, num_batch=num_batch,
                                                                     require_attr=False)
    else:

        return_results = generator.generate_completion_with_logits_K(K, model, prefix_batch, eval_completion_length, topk, topp,
                                                                                           batchsize=batchsize, num_batch=num_batch, require_attr=False)

    pred_completion, pred_completion_probs, max_completion, max_completion_probs = return_results
    pred_completion = pred_completion.cpu().tolist()
    pred_completion_probs = pred_completion_probs.cpu().tolist()
    max_completion = max_completion.cpu().tolist()
    max_completion_probs = max_completion_probs.cpu().tolist()

    # Calcuate unique n-grams / total grams metrics
    completion_metrics.update(pred_completion)

    completions.extend(pred_completion)
    completions_probs.extend(pred_completion_probs)

    completion_metrics = completion_metrics.report('generated')
    return completions, completion_metrics, completions_probs, max_completion, max_completion_probs

def single_token_prob_fetch(sentence, next_token_index, model, generator, topk, topp, beam_size, batchsize=1, num_batch=1):
    sample = {
        'id': torch.tensor([1]),
        'nsentences': 1,
        'ntokens': len(sentence),
        'net_input':{
            'src_tokens': torch.tensor(sentence).view(1, -1),
            'src_lengths': torch.tensor([len(sentence)])
        }
    }
    input_sequence = sample['net_input']['src_tokens']
    prefix_batch = input_sequence.cuda()

    target_prob, max_prob, max_token =  generator.generate_next_token_prob(model, prefix_batch, next_token_index)
    return target_prob.cpu().tolist(), max_prob.cpu().tolist(), max_token.cpu().tolist()


def batch_input_sequence_by_prefix_length(input_sequence, prefix_length):
    seq_len = input_sequence.size(1)
    # Discard tokens if the sequence length is not divisible by the prefix length.
    new_seq_len = (seq_len//prefix_length)*prefix_length
    input_sequence = input_sequence[:, :new_seq_len]
    batch = input_sequence.view(-1, prefix_length).contiguous()
    return batch


def batch_input_sequence_by_prefix_length_and_align_target(input_sequence, prefix_length, eval_completion_length):
    # Warning: The function can only be used for batchsize = 1 sentence.
    seq_len = input_sequence.size(1)
    # Discard tokens if the sequence length is not divisible by the prefix length.
    new_seq_len = (seq_len//prefix_length)*prefix_length
    input_sequence = input_sequence[:, :new_seq_len]
    batch = input_sequence.view(-1, prefix_length).contiguous()

    num_of_sentences = seq_len//prefix_length
    gt_comple = []
    for sentence_id in range(input_sequence.size(0)):
        for i in range(num_of_sentences):
            gt_comple.append(input_sequence[sentence_id, i * eval_completion_length : (i+1) * eval_completion_length].tolist())

    return batch, gt_comple


@torch.no_grad()
def single_token_prob_fetch_para(sentence, model, dictionary):
    sample = {
        'id': torch.tensor([1]),
        'nsentences': 1,
        'ntokens': len(sentence),
        'net_input': {
            'src_tokens': torch.tensor(sentence).view(1, -1),
            'src_lengths': torch.tensor([len(sentence)])
        }
    }

    sample = utils.move_to_cuda(sample)
    net_output = model(**sample['net_input'])
    logits = net_output[0][0]
    logits[:, dictionary.pad()] = -1e19
    probs = model.get_normalized_probs(net_output, log_probs=False)[0]
    predicted_max_probs, predicted_tokens = probs.max(1)
    predicted_max_probs = predicted_max_probs.tolist()
    predicted_tokens = predicted_tokens.tolist()

    target_sentence = sentence + [11]
    target_sentence = target_sentence[1:]
    target_sentence_cuda = torch.tensor(target_sentence).cuda()

    target_probs = torch.gather(probs, 1, target_sentence_cuda.view(-1, 1)).view(-1)
    target_probs = target_probs.tolist()

    return target_sentence, target_probs, predicted_tokens, predicted_max_probs



@torch.no_grad()
def eval_single_token_prediction(model, itr, dictionary, singletoken_topp=0.0, singletoken_topk=1):
    predicted_tokens = []
    target_tokens = []

    mle_loss_sum = 0
    num_samples_sum = 0
    wrong_mass_sum = 0

    logging_outputs = []

    for n, sample in tqdm(enumerate(itr)):
        sample = utils.move_to_cuda(sample)
        net_output = model(**sample['net_input'])
        logits = net_output[0][0]
        logits[:, dictionary.pad()] = -1e19
        predicted_tokens.append(logits.argmax(1).tolist())
        target = sample['target'].view(-1)
        target_tokens.append(target.tolist())

        # -- mle loss
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        true_token_lprobs = F.nll_loss(
            lprobs,
            target,
            ignore_index=dictionary.pad_index,
            reduction='none',
        )
        true_token_logits = -F.nll_loss(
            logits,
            target,
            ignore_index=dictionary.pad_index,
            reduction='none',
        )
        mle_loss = true_token_lprobs.sum()
        orig = utils.strip_pad(target, dictionary.pad_index)
        ntokens = orig.numel()

        mle_loss_sum += mle_loss.item()
        num_samples_sum += ntokens

        logging_output = TrainingMetrics.ranking_metrics(logits, true_token_logits, sample, ntokens, target, topk=singletoken_topk, topp=singletoken_topp)

        negative_targets = (logits > true_token_logits[:, None]).float()
        wrong_mass_sum += (negative_targets * (F.softmax(logits, dim=1))).sum()

        logging_outputs.append(logging_output)

    ppl = math.pow(2, mle_loss_sum / num_samples_sum / math.log(2))
    custom_metrics = TrainingMetrics.aggregate_and_normalize(logging_outputs)
    custom_metrics['ppl'] = ppl
    avg_wrong_mass = wrong_mass_sum / num_samples_sum
    custom_metrics['avg_wrong_mass'] = avg_wrong_mass.item()
    return predicted_tokens, target_tokens, custom_metrics

# ============ For evaluation_modify     =============
def generate_completions_m(model, generator, fairseq_generator, itr, eval_prefix_length, eval_completion_length, high_level_args, topk, topp, num_samples, beam_size, include_prefix=True):
    completions = []
    gt_completion = []
    completion_metrics = Metrics()
    actual_metrics = Metrics()
    for n, sample in enumerate(tqdm(itr)):
        input_sequence = sample['net_input']['src_tokens']
        prefix_batch, gt_com = batch_input_sequence_by_prefix_length_and_align_target(input_sequence, eval_prefix_length, eval_completion_length)
        prefix_batch = prefix_batch.cuda()
        if input_sequence.size(1) < eval_prefix_length:
            continue
        if beam_size > 1:
            assert topk == 1, 'with greedy topk must be 1'
            assert topp == 0.0, 'with greedy topp must be 0'
            sample['net_input']['src_tokens'] = prefix_batch
            res = fairseq_generator.generate([model], sample, prefix_batch, bos_token=0)  # prefix is there in preds!
            pred_completion = [res[i][0]['tokens'][eval_prefix_length:-1].cpu().tolist() for i in range(len(res))]
        elif beam_size == 1:
            if high_level_args.force_sample_top:
                pred_completion, _, _, _ = generator.generate_completion_with_logits_K(high_level_args.force_sample_top_k, model, prefix_batch, eval_completion_length, topk, topp)
            elif high_level_args.force_sample_top_truncate:
                pred_completion, _, _, _ = generator.generate_completion_with_logits_K_truncate(
                    2, high_level_args.force_sample_top_truncate_ratio, high_level_args.force_sample_top_truncate_K, high_level_args.force_sample_top_truncate_style,
                    model, prefix_batch, eval_completion_length, topk, topp)
            else:
                pred_completion = generator.generate_completion(model, prefix_batch, eval_completion_length, topk, topp)
            pred_completion = pred_completion.cpu().tolist()
        # Calcuate unique n-grams / total grams metrics
        completion_metrics.update(pred_completion)
        actual_metrics.update(input_sequence)

        if include_prefix:
            prefix_batch = prefix_batch.cpu().tolist()
            pred_completion = [prefix + completion for
                               prefix, completion in zip(prefix_batch, pred_completion)]
            gt_com = [prefix + completion for
                               prefix, completion in zip(prefix_batch, gt_com)]

        completions.extend(pred_completion)
        gt_completion.extend(gt_com)

        if n == num_samples:
            break

    completion_metrics = completion_metrics.report('generated')
    actual_metrics = actual_metrics.report('actual')
    return completions, gt_completion, completion_metrics, actual_metrics

# ============ For Intra sentence evalution ===========
def single_intra_completitions(sentence, model, generator, fairseq_generator, vocab, eval_prefix_length, eval_completion_length,
                       topk, topp, beam_size, batchsize=1, num_batch=1, require_attr=False):
    completions = []
    completions_probs = []
    completion_metrics = Metrics()
    sample = {
        'id': torch.tensor([1]),
        'nsentences': 1,
        'ntokens': len(sentence),
        'net_input': {
            'src_tokens': torch.tensor(sentence).view(1, -1),
            'src_lengths': torch.tensor([len(sentence)])
        }
    }
    input_sequence = sample['net_input']['src_tokens']
    prefix_batch = input_sequence.cuda()

    if beam_size > 1:
        assert topk == 1, 'with greedy topk must be 1'
        assert topp == 0.0, 'with greedy topp must be 0'
        sample['net_input']['src_tokens'] = prefix_batch
        res = fairseq_generator.generate([model], sample, prefix_batch, bos_token=0)  # prefix is there in preds!
        pred_completion = [res[i][0]['tokens'][eval_prefix_length:-1].cpu().tolist() for i in range(len(res))]
        pred_completion_probs = [res[i][0]['positional_scores'][eval_prefix_length:-1].exp().cpu().tolist() for i in
                                 range(len(res))]

        # Calcuate unique n-grams / total grams metrics
        completion_metrics.update(pred_completion)

        completions.extend(pred_completion)
        completions_probs.extend(pred_completion_probs)

        completion_metrics = completion_metrics.report('generated')
        return completions, completion_metrics, completions_probs, None

    elif beam_size == 1:

        return_results = generator.generate_intra_completion_with_logits(model, prefix_batch, eval_completion_length,
                                                                   topk, topp, vocab,
                                                                   batchsize=batchsize, num_batch=num_batch,
                                                                   require_attr=require_attr)
        all_attri_data = None
        if require_attr:
            pred_completion, pred_completion_probs, all_attri_data = return_results
        else:
            pred_completion, pred_completion_probs = return_results
        pred_completion = pred_completion.cpu().tolist()
        pred_completion_probs = pred_completion_probs.cpu().tolist()

        # Calcuate unique n-grams / total grams metrics
        completion_metrics.update(pred_completion)

        completions.extend(pred_completion)
        completions_probs.extend(pred_completion_probs)

        completion_metrics = completion_metrics.report('generated')
        return completions, completion_metrics, completions_probs, all_attri_data

def generate_intra_completions(model, generator, fairseq_generator, itr, vocab, eval_prefix_length, eval_completion_length, topk, topp, num_samples, beam_size, include_prefix=True):
    completions = []
    gt_completion = []
    completion_metrics = Metrics()
    actual_metrics = Metrics()
    for n, sample in enumerate(tqdm(itr)):
        input_sequence = sample['net_input']['src_tokens']
        prefix_batch, gt_com = batch_input_sequence_by_prefix_length_and_align_target(input_sequence,
                                                                                      eval_prefix_length,
                                                                                      eval_completion_length)
        prefix_batch = prefix_batch.cuda()
        if input_sequence.size(1) < eval_prefix_length:
            continue
        if beam_size > 1:
            assert 1 == 0, 'not implemented yet'
            assert topk == 1, 'with greedy topk must be 1'
            assert topp == 0.0, 'with greedy topp must be 0'
            sample['net_input']['src_tokens'] = prefix_batch
            res = fairseq_generator.generate([model], sample, prefix_batch, bos_token=0)  # prefix is there in preds!
            pred_completion = [res[i][0]['tokens'][eval_prefix_length:-1].cpu().tolist() for i in range(len(res))]
        elif beam_size == 1:
            pred_completion = generator.generate_intra_completion(model, vocab, prefix_batch, eval_completion_length, topk, topp)
            pred_completion = pred_completion.cpu().tolist()
        # Calcuate unique n-grams / total grams metrics
        completion_metrics.update(pred_completion)
        actual_metrics.update(input_sequence)

        if include_prefix:
            prefix_batch = prefix_batch.cpu().tolist()
            pred_completion = [prefix + completion for
                               prefix, completion in zip(prefix_batch, pred_completion)]
            gt_com = [prefix + completion for
                      prefix, completion in zip(prefix_batch, gt_com)]
        completions.extend(pred_completion)
        gt_completion.extend(gt_com)

        if n == num_samples:
            break

    completion_metrics = completion_metrics.report('generated')
    actual_metrics = actual_metrics.report('actual')
    return completions, gt_completion, completion_metrics, actual_metrics

def single_completions_intra_with_k_candidate(sentence, model, generator, K, vocab, eval_prefix_length, eval_completion_length, topk, topp, beam_size, batchsize=1, num_batch=1, require_attr=False):
    completions = []
    completions_probs = []
    completion_metrics = Metrics()
    sample = {
        'id': torch.tensor([1]),
        'nsentences': 1,
        'ntokens': len(sentence),
        'net_input':{
            'src_tokens': torch.tensor(sentence).view(1, -1),
            'src_lengths': torch.tensor([len(sentence)])
        }
    }
    input_sequence = sample['net_input']['src_tokens']
    prefix_batch = input_sequence.cuda()


    return_results = generator.generate_intra_completion_with_logits_K(K, model, prefix_batch, vocab, eval_completion_length, topk, topp,
                                                                                       batchsize=batchsize, num_batch=num_batch, require_attr=False)

    pred_completion, pred_completion_probs, max_completion, max_completion_probs = return_results
    pred_completion = pred_completion.cpu().tolist()
    pred_completion_probs = pred_completion_probs.cpu().tolist()
    max_completion = max_completion.cpu().tolist()
    max_completion_probs = max_completion_probs.cpu().tolist()

    # Calcuate unique n-grams / total grams metrics
    completion_metrics.update(pred_completion)

    completions.extend(pred_completion)
    completions_probs.extend(pred_completion_probs)

    completion_metrics = completion_metrics.report('generated')
    return completions, completion_metrics, completions_probs, max_completion, max_completion_probs


def _intra_sentence_mask(source, vocab):
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

def _gen_intra_sentence_mask(tokens, vocab, last=True):
    batch_size = tokens.size()[0]
    o_tokens = tokens.cpu()

    masks = []
    for idx in range(batch_size):
        mask = _intra_sentence_mask(o_tokens[idx], vocab)
        masks.append(mask)
    masks = torch.stack(masks, dim=0)
    if last:
        masks = masks[:, -1:]

    masks = fill_with_neg_inf(torch.zeros(masks.size())).masked_fill_(masks, 0)
    masks = masks.to(device=tokens.device)

    return masks

@torch.no_grad()
def single_token_prob_intra_sentence_fetch_para(sentence, model, dictionary):
    src_tokens = torch.tensor(sentence).view(1, -1)
    src_attn_mask = _gen_intra_sentence_mask(src_tokens, dictionary, last=False)
    sample = {
        'id': torch.tensor([1]),
        'nsentences': 1,
        'ntokens': len(sentence),
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': torch.tensor([len(sentence)]),
            'src_attn_mask': src_attn_mask
        }
    }

    sample = utils.move_to_cuda(sample)
    net_output = model(**sample['net_input'])
    logits = net_output[0][0]
    logits[:, dictionary.pad()] = -1e19
    probs = model.get_normalized_probs(net_output, log_probs=False)[0]
    predicted_max_probs, predicted_tokens = probs.max(1)
    predicted_max_probs = predicted_max_probs.tolist()
    predicted_tokens = predicted_tokens.tolist()

    target_sentence = sentence + [11]
    target_sentence = target_sentence[1:]
    target_sentence_cuda = torch.tensor(target_sentence).cuda()

    target_probs = torch.gather(probs, 1, target_sentence_cuda.view(-1, 1)).view(-1)
    target_probs = target_probs.tolist()

    return target_sentence, target_probs, predicted_tokens, predicted_max_probs

# ========END======== For Intra sentence evalution ===========
# ========START ======== For see more candidates's prob for more insights
def single_completions_with_see_k_candidate(sentence, model, generator, K, eval_prefix_length, eval_completion_length, topk, topp, beam_size, batchsize=1, num_batch=1, require_attr=False, show_K=4):
    completions = []
    completions_probs = []
    completion_metrics = Metrics()
    sample = {
        'id': torch.tensor([1]),
        'nsentences': 1,
        'ntokens': len(sentence),
        'net_input':{
            'src_tokens': torch.tensor(sentence).view(1, -1),
            'src_lengths': torch.tensor([len(sentence)])
        }
    }
    input_sequence = sample['net_input']['src_tokens']
    prefix_batch = input_sequence.cuda()

    return_results = generator.generate_completion_with_more_logits(K, show_K, model, prefix_batch, eval_completion_length, topk, topp,
                                                                                           batchsize=batchsize, num_batch=num_batch, require_attr=False)

    pred_completion, pred_completion_probs, max_completion, max_completion_probs, aj_similarity = return_results
    pred_completion = pred_completion.cpu().tolist()
    pred_completion_probs = pred_completion_probs.cpu().tolist()

    max_completion = max_completion.cpu().numpy()
    max_completion_probs = max_completion_probs.cpu().numpy()
    aj_similarity = aj_similarity.cpu().numpy()

    # Calcuate unique n-grams / total grams metrics
    completion_metrics.update(pred_completion)

    completions.extend(pred_completion)
    completions_probs.extend(pred_completion_probs)

    completion_metrics = completion_metrics.report('generated')
    return completions, completion_metrics, completions_probs, max_completion, max_completion_probs, aj_similarity