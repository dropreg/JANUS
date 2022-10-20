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

@register_criterion("janus_loss")
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
        
    def forward_single(self, model, sample, reduce=True):
        
        nar_loss, nar_nll_loss = self.compute_nar_loss(model, sample)
        ar_loss, ar_nll_loss = self.compute_ar_loss(model, sample)
        
        loss = 0.3 * ar_loss + 0.7 * nar_loss
        
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "ar_loss": ar_loss.data,
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
        content_out, query_out, length_out, length_tgt, nar_states = model(**net_input)
        
        content_m = ~sample["nar_input"]["nar_content_m"]
        query_m = sample["nar_input"]["nar_query_w"].ne(self.padding_idx)
        target = sample["nar_input"]["nar_target"]
        target_mask = target.ne(self.padding_idx)

        content_loss, content_nll_loss = self.label_smooth_loss(content_out[content_m], target[target_mask])
        query_loss, query_nll_loss = self.label_smooth_loss(query_out[query_m], target[target_mask])
        length_loss = self.length_loss(length_out, length_tgt)
        
        loss = 0.5 * (query_loss + content_loss) + 0.1 * length_loss
        nll_loss = query_nll_loss + content_nll_loss

        content_hidden = nar_states["content_state"][content_m].unsqueeze(1)
        query_hidden = nar_states["query_state"][query_m].unsqueeze(1)
        
        hidden_state = torch.cat([content_hidden, query_hidden], dim=1)
        ctl_loss = self.contrastive_loss(hidden_state, 0.5)
        
        return loss, nll_loss

    def label_smooth_loss(self, net_out, net_target):
        net_logits = F.log_softmax(net_out, dim=-1)
        nll_loss = F.nll_loss(net_logits, net_target, reduction="none").float().mean()
        loss = nll_loss * (1. - self.eps) - net_logits.float().mean() * self.eps
        return loss, nll_loss

    def length_loss(self, length_out, length_tgt):
        length_logits = F.log_softmax(length_out, dim=-1)
        length_loss = F.nll_loss(length_logits, length_tgt, reduction="none").float().mean()
        return length_loss
        
    def contrastive_loss(self, last_hidden_states, margin):
        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        score_matrix = torch.matmul(norm_rep, norm_rep.transpose(1,2)) 

        bsz, seqlen, _ = score_matrix.size()
        gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) # bsz x seqlen
        gold_score = torch.unsqueeze(gold_score, -1)
        assert gold_score.size() == torch.Size([bsz, seqlen, 1])
        difference_matrix = gold_score - score_matrix
        assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
        loss_matrix = margin - difference_matrix
        loss_matrix = torch.nn.functional.relu(loss_matrix)
        cl_loss = torch.mean(loss_matrix.sum(-1))
        return cl_loss

      
    ######################## kl Loss  ##########################
    def build_concat_ar_data(self, sample):
        ar_input ={
            "src_tokens": torch.cat([sample["net_input"]["src_tokens"], 
                                     sample["net_input"]["src_tokens"].clone()], dim=0),
            "src_lengths": torch.cat([sample["net_input"]["src_lengths"], 
                                     sample["net_input"]["src_lengths"].clone()], dim=0),
            "ar_target": torch.cat([sample["ar_input"]["ar_target"], 
                                     sample["ar_input"]["ar_target"].clone()], dim=0),
            "ar_content_w": torch.cat([sample["ar_input"]["ar_content_w"], 
                                     sample["ar_input"]["ar_content_w"].clone()], dim=0),
            "mode": 'ar',
        }
        return ar_input

    def build_concat_nar_data(self, sample):
        nar_input ={
            "src_tokens": torch.cat([sample["net_input"]["src_tokens"],
                                     sample["net_input"]["src_tokens"].clone()], dim=0),
            "src_lengths": torch.cat([sample["net_input"]["src_lengths"], 
                                     sample["net_input"]["src_lengths"].clone()], dim=0),
            "nar_content_w": torch.cat([sample["nar_input"]["nar_content_w"], 
                                     sample["nar_input"]["nar_content_w"].clone()], dim=0),
            "nar_content_p": torch.cat([sample["nar_input"]["nar_content_p"], 
                                     sample["nar_input"]["nar_content_p"].clone()], dim=0),
            "nar_query_w": torch.cat([sample["nar_input"]["nar_query_w"], 
                                     sample["nar_input"]["nar_query_w"].clone()], dim=0),
            "nar_query_p": torch.cat([sample["nar_input"]["nar_query_p"], 
                                     sample["nar_input"]["nar_query_p"].clone()], dim=0),
            "target": torch.cat([sample["target"], sample["target"].clone()], dim=0),
            "mode": 'nar',
        }
        return nar_input

    def forward(self, model, sample, reduce=True):
        
        nar_loss, nar_nll_loss = self.compute_nar_kl_loss(model, sample)
        # ar_loss, ar_nll_loss = self.compute_ar_kl_loss(model, sample)
        
        loss = nar_loss
        
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "ar_loss": nar_nll_loss.data,
            "nar_loss": nar_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_kl_loss(self, outputs):
        outputs_logprob = F.log_softmax(outputs.float(), dim=-1)
        outputs_prob = F.softmax(outputs.float(), dim=-1)

        p, q = torch.split(outputs_logprob, outputs_logprob.size(0)//2, dim=0)
        p_tec, q_tec = torch.split(outputs_prob, outputs_logprob.size(0)//2, dim=0)

        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none').sum(-1)
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none').sum(-1)

        loss = (p_loss + q_loss) / 2
        return loss.mean()
    
    def compute_ar_kl_loss(self, model, sample):
        
        net_input = self.build_concat_ar_data(sample)
        ar_out = model(**net_input)

        target = torch.cat([sample["ar_input"]["ar_target"], 
                            sample["ar_input"]["ar_target"].clone()], dim=0)
        mask = target.ne(self.padding_idx)
        loss, nll_loss = self.label_smooth_loss(ar_out[mask], target[mask])

        kl_loss = self.compute_kl_loss(ar_out[mask])
        loss += kl_loss
        return  loss, nll_loss

    def compute_nar_kl_loss(self, model, sample):
        
        net_input = self.build_concat_nar_data(sample)
        content_out, query_out, length_out, length_tgt, nar_states = model(**net_input)

        content_m = torch.cat([~sample["nar_input"]["nar_content_m"],
                                ~sample["nar_input"]["nar_content_m"].clone()], dim=0)
        query_m = torch.cat([sample["nar_input"]["nar_query_w"],
                             sample["nar_input"]["nar_query_w"].clone()], dim=0).ne(self.padding_idx)
        target = torch.cat([sample["nar_input"]["nar_target"], 
                            sample["nar_input"]["nar_target"].clone()], dim=0)
        target_mask = target.ne(self.padding_idx)

        content_loss, content_nll_loss = self.label_smooth_loss(content_out[content_m], target[target_mask])
        query_loss, query_nll_loss = self.label_smooth_loss(query_out[query_m], target[target_mask])
        length_loss = self.length_loss(length_out, length_tgt)
        
        loss = query_loss + content_loss + 0.1 * length_loss
        nll_loss = query_nll_loss + content_nll_loss
        
        content_kl_loss = self.compute_kl_loss(content_out[content_m])
        query_kl_loss = self.compute_kl_loss(query_out[query_m])
        loss += content_kl_loss + query_kl_loss
        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ar_loss_sum = sum(log.get("ar_loss", 0) for log in logging_outputs)
        nar_loss_sum = sum(log.get("nar_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
