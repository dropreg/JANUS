# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils
from fairseq.utils import new_arange

def collate(
    samples,
    pad_idx,
    eos_idx,
    vocab,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
):
    assert input_feeding
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=None,  # use eos_idx of each sample instead of vocab.eos()
            left_pad=left_pad,
            move_eos_to_beginning=move_eos_to_beginning,
            pad_to_length=pad_to_length,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    nar_content_w = None
    nar_content_p = None
    nar_content_m = None
    nar_query_w = None
    nar_query_p = None
    nar_target = None

    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s["target"]) for s in samples)

        # NAR content
        nar_content_w = merge(
            "nar_content_w",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["nar_content_w"]
            if pad_to_length is not None
            else None,
        )
        nar_content_w = nar_content_w.index_select(0, sort_order)
        nar_content_p = merge(
            "nar_content_p",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["nar_content_p"]
            if pad_to_length is not None
            else None,
        )
        nar_content_p = nar_content_p.index_select(0, sort_order)
        nar_content_m = merge(
            "nar_content_m",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["nar_content_m"]
            if pad_to_length is not None
            else None,
        )
        nar_content_m = nar_content_m.index_select(0, sort_order)
        nar_query_w = merge(
            "nar_query_w",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["nar_query_w"]
            if pad_to_length is not None
            else None,
        )
        nar_query_w = nar_query_w.index_select(0, sort_order)
        nar_query_p = merge(
            "nar_query_p",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["nar_query_p"]
            if pad_to_length is not None
            else None,
        )
        nar_query_p = nar_query_p.index_select(0, sort_order)
        nar_target = merge(
            "nar_target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["nar_target"]
            if pad_to_length is not None
            else None,
        )
        nar_target = nar_target.index_select(0, sort_order)
        ar_mask = merge(
            "ar_mask",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["ar_mask"]
            if pad_to_length is not None
            else None,
        )
        ar_mask = ar_mask.index_select(0, sort_order)
        
        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
            # mask_prev_output_tokens = merge(
            #     "target_in",
            #     left_pad=left_pad_target,
            #     pad_to_length=pad_to_length["target_in"]
            #     if pad_to_length is not None
            #     else None,
            # )
            # mask_prev_output_tokens = mask_prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s["source"]) for s in samples)
    
    batch = {
        "id": id,
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
        "nsentences": samples[0]["source"].size(0),
        "sort_order": sort_order,
    }

    if prev_output_tokens is not None:
        batch["ar_input"] = {
            "ar_target": target,
            "ar_content_w": prev_output_tokens,
            "ar_mask": ar_mask,
        }
        # batch["net_input"]["mask_prev_output_tokens"] = mask_prev_output_tokens
        # batch["mask"] = target != mask_prev_output_tokens
        batch["nar_input"] = {
            "nar_content_w": nar_content_w,
            "nar_content_p": nar_content_p,
            "nar_content_m": nar_content_m,
            "nar_query_w": nar_query_w,
            "nar_query_p": nar_query_p,
            "nar_target": nar_target,
        }
        
    return batch


class DenoisingSpanDataset(FairseqDataset):
    """
    A wrapper around TokenBlockDataset for BART dataset.

    Args:
        dataset (TokenBlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        mask_idx (int): dictionary index used for masked token
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
        seed: Seed for random number generator for reproducibility.
        args: argparse arguments.
    """

    def __init__(
        self,
        dataset,
        sizes,
        vocab,
        mask_idx,
        mask_whole_words,
        shuffle,
        seed,
        args,
        eos=None,
        item_transform_func=None,
    ):
        self.dataset = dataset

        self.sizes = sizes

        self.vocab = vocab
        self.shuffle = shuffle
        self.seed = seed
        self.mask_idx = mask_idx
        self.mask_whole_word = mask_whole_words
        self.mask_ratio = args.mask
        self.random_ratio = args.mask_random
        self.insert_ratio = args.insert
        self.rotate_ratio = args.rotate
        self.permute_sentence_ratio = args.permute_sentences
        self.eos = eos if eos is not None else vocab.eos()
        self.unk = vocab.unk()
        self.item_transform_func = item_transform_func

        if args.bpe != "gpt2":
            self.full_stop_index = self.vocab.eos()
        else:
            assert args.bpe == "gpt2"
            self.full_stop_index = self.vocab.index("13")

        self.replace_length = args.replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if args.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={args.mask_length}")
        if args.mask_length == "subword" and args.replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace-length=1 or 0")

        self.mask_span_distribution = None
        if args.mask_length == "span-poisson":
            _lambda = args.poisson_lambda

            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        self.epoch = 0

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True  # only the noise changes, not item sizes

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def filter_indices_by_size(self, indices, max_sizes):
        # double max_sizes
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif (
                hasattr(self, "sizes")
                and isinstance(self.sizes, list)
                and len(self.sizes) == 1
            ):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = data_utils._filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = data_utils._filter_by_size_dynamic(
                indices, self.size, max_sizes
            )
        return indices, ignored

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            
            tokens = self.dataset[index]
            assert tokens[-1] == self.eos
            source, target = tokens, tokens.clone()

            if self.mask_ratio > 0:
                source, decoder_mask, predict_mask = self.add_whole_word_mask(source, self.mask_ratio)
                target_mask = decoder_mask | predict_mask
                target_in = target.masked_fill(target_mask, self.unk)

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                source = self.add_rolling_noise(source)
            
        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos()
        assert source[-1] == self.eos

        nar_target = target.clone()
        masked_nar_sent = target_in.clone()
        nar_mask = target_mask
        ar_target = target
        position = new_arange(target)
        
        shuffle_context = False
        if shuffle_context:
            shuffle_order = torch.randperm(nar_target[nar_mask].size(0))
            nar_content_w = torch.cat([masked_nar_sent[~nar_mask], 
                                    masked_nar_sent[nar_mask][shuffle_order],
                                    nar_target[nar_mask][shuffle_order]], dim=0)
            nar_content_p = torch.cat([position[~nar_mask],
                                    position[nar_mask][shuffle_order],
                                    position[nar_mask][shuffle_order]], dim=0)
            nar_content_m = torch.cat([~nar_mask[~nar_mask], ~nar_mask[nar_mask],
                                    nar_mask[nar_mask]], dim=0)
            nar_query_w = masked_nar_sent[nar_mask][shuffle_order]
            nar_query_p = position[nar_mask][shuffle_order]
            nar_target = nar_target[nar_mask][shuffle_order]
        else:
            nar_content_w = torch.cat([masked_nar_sent[~nar_mask],
                            masked_nar_sent[nar_mask], nar_target[nar_mask]], dim=0)
            nar_content_p = torch.cat([position[~nar_mask],
                            position[nar_mask], position[nar_mask]], dim=0)
            nar_content_m = torch.cat([~nar_mask[~nar_mask], 
                            ~nar_mask[nar_mask], nar_mask[nar_mask]], dim=0)
            nar_query_w = masked_nar_sent[nar_mask]
            nar_query_p = position[nar_mask]
            nar_target = nar_target[nar_mask]
            
        return {
            "id": index,
            "source": source,
            "target": target,
            "ar_mask": ~nar_mask, 
            "nar_content_w": nar_content_w,
            "nar_content_p": nar_content_p,
            "nar_content_m": nar_content_m,
            "nar_content_w": nar_content_w,
            "nar_query_w": nar_query_w,
            "nar_query_p": nar_query_p,
            "nar_target": nar_target,
        }

    def __len__(self):
        return len(self.dataset)

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p):
        recover_source = source.clone()
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)
            
            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < 0.5

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[ -1] = 255  
        # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices[~mask_random]] = self.mask_idx
            source[indices[mask_random]] = -100

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices[~mask_random]] = 0
                    source[indices[~mask_random]] = self.mask_idx
                    source[indices[mask_random]] = -100
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx

                assert source_length - 1 not in indices    
        
        decoder_mask = source.eq(-100)
        predict_mask = source.eq(self.mask_idx)
        recover_mask = decoder_mask
        source[recover_mask] = recover_source[recover_mask]
        source = source[to_keep]
        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))
        return source, decoder_mask, predict_mask


    def add_permuted_noise(self, tokens, p):
        num_words = len(tokens)
        num_to_permute = math.ceil(((num_words * 2) * p) / 2.0)
        substitutions = torch.randperm(num_words - 2)[:num_to_permute] + 1
        tokens[substitutions] = tokens[substitutions[torch.randperm(num_to_permute)]]
        return tokens

    def add_rolling_noise(self, tokens):
        offset = np.random.randint(1, max(1, tokens.size(-1) - 1) + 1)
        tokens = torch.cat(
            (tokens[0:1], tokens[offset:-1], tokens[1:offset], tokens[-1:]),
            dim=0,
        )
        return tokens

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=1, high=len(self.vocab), size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch of data
        """
        return collate(
            samples, self.vocab.pad(), self.eos, self.vocab, pad_to_length=pad_to_length
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.sizes[indices], kind="mergesort")]

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.src, "supports_prefetch")
            and self.src.supports_prefetch
            and hasattr(self.tgt, "supports_prefetch")
            and self.tgt.supports_prefetch
        )
