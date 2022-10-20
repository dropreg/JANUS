# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions.masked_lm import MaskedLmLoss
from fairseq.criterions import register_criterion
    
@register_criterion("janus_pretrain_loss")
class JanusPretrainLoss(MaskedLmLoss):

    def build_ar_data(self, sample):
        ar_input ={
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths": sample["net_input"]["src_lengths"],
            "ar_target": sample["ar_input"]["ar_target"],
            "ar_content_w": sample["ar_input"]["ar_content_w"],
            "mode": 'ar',
        }
        return ar_input

    def build_nar_data(self, sample):
        nar_input ={
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths": sample["net_input"]["src_lengths"],
            "nar_content_w": sample["nar_input"]["nar_content_w"],
            "nar_content_p": sample["nar_input"]["nar_content_p"],
            "nar_query_w": sample["nar_input"]["nar_query_w"],
            "nar_query_p": sample["nar_input"]["nar_query_p"],
            "target": sample["target"],
            "mode": 'nar',
        }
        return nar_input

    def forward(self, model, sample, reduce=True):
        ar_net_input = self.build_ar_data(sample)
        ar_out = model(**ar_net_input)

        ar_target = sample["ar_input"]["ar_target"]
        mask = ar_target.ne(self.padding_idx)
        ar_loss = self.label_smooth_loss(ar_out[mask], ar_target[mask])

        nar_net_input = self.build_nar_data(sample)
        content_out, query_out = model(**nar_net_input)
        
        content_m = ~sample["nar_input"]["nar_content_m"]
        query_m = sample["nar_input"]["nar_query_w"].ne(self.padding_idx)
        nar_target = sample["nar_input"]["nar_target"]
        target_mask = nar_target.ne(self.padding_idx)
        
        content_loss = self.label_smooth_loss(content_out[content_m], nar_target[target_mask])
        query_loss = self.label_smooth_loss(query_out[query_m], nar_target[target_mask])
        nar_loss = query_loss + content_loss
        
        loss = 0.5 * (ar_loss + nar_loss)
        
        nar_kl_loss = self.compute_cross_kl_loss(content_out[content_m], query_out[query_m])        
        ar_distill_mask = ~sample['ar_input']["ar_mask"]
        ar_kl_loss = self.compute_cross_kl_loss(ar_out[ar_distill_mask], query_out[query_m])
        loss += 0.5 * (nar_kl_loss + ar_kl_loss)

        sample_size = sample["ntokens"]
        logging_output = {
            "loss": loss.data,
            "ar_loss": nar_loss.data,
            "nar_loss": nar_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_cross_kl_loss(self, content_outputs, query_outputs):
        p = F.log_softmax(content_outputs.float(), dim=-1)
        q = F.log_softmax(query_outputs.float(), dim=-1)
        p_tec = F.softmax(content_outputs.float(), dim=-1)
        q_tec = F.softmax(query_outputs.float(), dim=-1)

        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')

        loss = (p_loss + q_loss) / 2
        return loss.sum()

    def forward_single(self, model, sample, reduce=True):
        
        nar_loss = self.compute_nar_loss(model, sample)
        # ar_loss, ar_nll_loss = self.compute_ar_loss(model, sample)
        loss = nar_loss
        
        sample_size = sample["ntokens"]
        logging_output = {
            "loss": loss.data,
            "ar_loss": nar_loss.data,
            "nar_loss": nar_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_ar_loss(self, model, sample):
        
        net_input = self.build_ar_data(sample)
        ar_out = model(**net_input)

        target = sample["ar_input"]["ar_target"]
        mask = target.ne(self.padding_idx)
        loss, nll_loss = self.label_smooth_loss(ar_out[mask], target[mask])
        return  loss, nll_loss

    def compute_nar_loss(self, model, sample):
        
        net_input = self.build_nar_data(sample)
        content_out, query_out = model(**net_input)
        
        content_m = ~sample["nar_input"]["nar_content_m"]
        query_m = sample["nar_input"]["nar_query_w"].ne(self.padding_idx)
        target = sample["nar_input"]["nar_target"]
        target_mask = target.ne(self.padding_idx)
        
        content_loss = self.label_smooth_loss(content_out[content_m], target[target_mask])
        query_loss = self.label_smooth_loss(query_out[query_m], target[target_mask])

        loss = query_loss + content_loss
        return loss

    def label_smooth_loss(self, net_out, net_target):
        net_logits = F.log_softmax(net_out, dim=-1)
        loss = F.nll_loss(net_logits, net_target, reduction="none").float().sum()
        return loss

    def length_loss(self, length_out, length_tgt):
        length_logits = F.log_softmax(length_out, dim=-1)
        length_loss = F.nll_loss(length_logits, length_tgt, reduction="none").float().mean()
        return length_loss
        
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ar_loss_sum = sum(log.get("ar_loss", 0) for log in logging_outputs)
        nar_loss_sum = sum(log.get("nar_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ar_loss", ar_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nar_loss", nar_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )