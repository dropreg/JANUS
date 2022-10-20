# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import torch.nn.functional as F

@register_criterion("ar_loss")
class JanusLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
    ):
        super().__init__(task)
        self.eps = 0.1

    def build_ar_data(self, sample):
        ar_input ={
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths": sample["net_input"]["src_lengths"],
            "ar_target": sample["ar_input"]["ar_target"],
            "ar_content_w": sample["ar_input"]["ar_content_w"],
            "mode": 'ar',
        }
        return ar_input
        
    def forward(self, model, sample, reduce=True):
        loss, nll_loss = self.compute_ar_loss(model, sample)        
        
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output
    
    def label_smooth_loss(self, net_out, net_target):
        net_logits = F.log_softmax(net_out, dim=-1)
        nll_loss = F.nll_loss(net_logits, net_target, reduction="none").float().mean()
        loss = nll_loss * (1. - self.eps) - net_logits.float().mean() * self.eps
        return loss, nll_loss

    def compute_ar_loss(self, model, sample):
        
        net_input = self.build_ar_data(sample)
        ar_out = model(**net_input)

        target = sample["ar_input"]["ar_target"]
        mask = target.ne(self.padding_idx)
        loss, nll_loss = self.label_smooth_loss(ar_out[mask], target[mask])
        return  loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
