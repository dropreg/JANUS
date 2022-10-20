# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""
from typing import Optional

import logging

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerDecoder, TransformerEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import random
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import Tensor
from .hub_interface import EncDecHubInterface
from fairseq.models.bart import BARTModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import torch.nn.functional as F
from fairseq import utils
from fairseq.models.nat import ensemble_decoder, ensemble_encoder
from fairseq.models.transformer import Embedding
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.utils import new_arange

from fairseq.models.transformer import TransformerEncoder
from .janus_trans_layers import JanusTransformerDecoderLayer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)

logger = logging.getLogger(__name__)


@register_model("janus_bart")
class JanusBARTModel(BARTModel):
    __jit_unused_properties__ = ["supported_targets"]

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()

        self.tgt_dict = decoder.dictionary
        self.bos = decoder.dictionary.bos()
        self.eos = decoder.dictionary.eos()
        self.pad = decoder.dictionary.pad()
        self.unk = decoder.dictionary.unk()

    @staticmethod
    def add_args(parser):
        super(JanusBARTModel, JanusBARTModel).add_args(parser)
        parser.add_argument(
            "--apply-bert-init",
            action="store_true",
            help="use custom param initialization for BERT",
        )

    @property
    def allow_length_beam(self):
        return True

    @property
    def allow_ensemble(self):
        return True

    def enable_ensemble(self, models):
        self.encoder.ensemble_models = [m.encoder for m in models]
        self.decoder.ensemble_models = [m.decoder for m in models]

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = JanusTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(self, **sample_input):
        
        src_tokens = sample_input["src_tokens"]
        src_lengths = sample_input["src_lengths"]
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)

        if sample_input["mode"] == 'ar':
            prev_output_tokens = sample_input["ar_content_w"]
            return self.forward_ar(prev_output_tokens, encoder_out, src_lengths)
        elif sample_input["mode"] == 'nar':
            nar_content_w = sample_input["nar_content_w"]
            nar_content_p = sample_input["nar_content_p"]
            nar_query_w = sample_input["nar_query_w"]
            nar_query_p = sample_input["nar_query_p"]
            target = sample_input["target"]
            return self.forward_nar(encoder_out, nar_content_w, 
                    nar_content_p, nar_query_w, nar_query_p, target)
        elif sample_input["mode"] == 'mix':
            pass
            
    def forward_ar(self, prev_output_tokens, encoder_out, src_lengths):
        ar_out, _ = self.decoder.forward_ar(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
        )
        return ar_out

    def forward_nar(self, encoder_out, nar_content_w, nar_content_p, nar_query_w, nar_query_p, target):

        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out, target
        )
        nar_content_out, nar_query_out, nar_states = self.decoder.forward_nar(
            normalize=False,
            encoder_out=encoder_out,
            content_w=nar_content_w,
            content_p=nar_content_p,
            query_w=nar_query_w,
            query_p=nar_query_p,
        )
        return nar_content_out, nar_query_out, length_out, length_tgt, nar_states

    def forward_mix(self, sample_input):
        pass
    
    def forward_encoder(self, encoder_inputs):
        return self.encoder(*encoder_inputs)
    #################################   NAR   #####################################
    @property
    def allow_length_beam(self):
        return True

    def forward_nar_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        _scores, _tokens = self.decoder.forward_nar(
            normalize=True,
            content_w=output_tokens,
            encoder_out=encoder_out,
        )[0].max(-1)

        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, src_tokens):
        # length prediction
        length_tgt = self.decoder.forward_length_prediction(
            self.decoder.forward_length(normalize=True, encoder_out=encoder_out),
            encoder_out=encoder_out,
        )

        max_length = length_tgt.clamp_(min=2).max()
        idx_length = utils.new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        # state_dict["decoder.embed_length.weight"] = self.decoder.embed_length.weight
        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
            self.encoder.dictionary
        ):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
                -1, :
            ]

            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["encoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["decoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
        
        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

class JanusTransformerDecoder(FairseqIncrementalDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.unk_idx = dictionary.unk()
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

        ################################################################
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.embed_length = Embedding(256, args.encoder_embed_dim, None)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = JanusTransformerDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        return layer

    ################################### FOR AR MODE #######################################
    def forward_ar(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        src_lengths: Optional[Any] = None,
    ):
        x, extra = self.extract_features_ar(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features_ar(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        return self.extract_features_ar_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
        )

    def make_ar_query_stream(self, prev_output_tokens):
        query_padding_mask = prev_output_tokens.eq(self.padding_idx)
        query_token = prev_output_tokens.clone().masked_fill(~query_padding_mask, self.unk_idx)
        return query_token
        
    def extract_features_ar_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        
        query_x = self.make_ar_query_stream(prev_output_tokens)
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        query_x = self.embed_scale * self.embed_tokens(query_x)

        if self.quant_noise is not None:
            x = self.quant_noise(x)
            query_x = self.quant_noise(query_x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
            query_x = self.project_in_dim(query_x)

        if positions is not None:
            x += positions
            query_x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
            query_x = self.layernorm_embedding(query_x)

        x = self.dropout_module(x)
        query_x = self.dropout_module(query_x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        query_x = query_x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        
        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, query_x, layer_attn, _ = layer.forward_ar(
                x,
                query_x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
            inner_states.append(x)
            if layer_attn is not None:
                attn = layer_attn.float().to(x)

        if attn is not None:
            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            query_x = self.layer_norm(query_x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        query_x = query_x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
            query_x = self.project_out_dim(query_x)

        x = query_x
        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]
    
    ################################### FOR NAR MODE #######################################
    def forward_nar(self, 
            normalize,
            encoder_out,
            content_w,
            content_p=None,
            query_w=None, 
            query_p=None, 
            step=0
        ):
        
        content_features, query_features = self.extract_features_nar(
            content_w,
            content_p,
            query_w,
            query_p,
            encoder_out=encoder_out,
        )
        content_out = self.output_layer(content_features)
        content_out = F.log_softmax(content_out, -1) if normalize else content_out
        if query_features is not None:
            query_out = self.output_layer(query_features)
            query_out = F.log_softmax(query_out, -1) if normalize else query_out
        else:
            query_out = None
        state_dict = {"content_state": content_features, "query_state": query_features}
        return content_out, query_out, state_dict

    def forward_length(self, normalize, encoder_out):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None
        enc_feats = _mean_pooling(enc_feats, src_masks)
        # enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def extract_features_nar(
        self,
        content_w=None,
        content_p=None,
        query_w=None,
        query_p=None,
        encoder_out=None,
    ):
        # embedding
        x, content_padding_mask, query_x, query_padding_mask = \
            self.forward_nar_embedding(content_w, content_p, query_w, query_p)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        if query_x is not None:
            query_x = query_x.transpose(0, 1)
            content_padding_mask, query_padding_mask = \
                self.make_query_content_mask_cuda(content_padding_mask, query_padding_mask)
        
        # decoder layers
        for i, layer in enumerate(self.layers):

            x, query_x, _, _ = layer.forward_nar(
                x,
                query_x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                content_padding_mask=content_padding_mask,
                query_padding_mask=query_padding_mask,
            )

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if query_x is not None:
            if self.layer_norm:
                query_x = self.layer_norm(query_x)
            query_x = query_x.transpose(0, 1)
            if self.project_out_dim is not None:
                query_x = self.project_out_dim(query_x)

        return x, query_x

    def make_query_content_mask_cuda(self, content_padding_mask, query_padding_mask):
        bsz, query_len = query_padding_mask.size()
        bsz, content_len = content_padding_mask.size()
        batch_b = (~query_padding_mask).sum(-1)
        batch_a = (~content_padding_mask).sum(-1)

        content_mask = []
        query_mask = []
        device = query_padding_mask.device
        temp_cache = {}

        for bsz_idx in range(bsz):
            b = batch_b[bsz_idx]
            a = batch_a[bsz_idx]
            key = (a.item(), b.item())

            if key in temp_cache:
                content_mask.append(temp_cache[key][0].clone())
                query_mask.append(temp_cache[key][1].clone())
            else:
                mask = torch.zeros((a - b , b), device=device)
                mask_tril = torch.tril(torch.ones((b, b), device=device), 0)
                c_mask = torch.cat([mask, mask_tril], dim=0)
                c_right_mask = torch.cat([c_mask, 1 - c_mask], dim=-1)
                c_left_mask = torch.zeros((a , a - 2 * b), device=device)
                c_all_mask = torch.cat([c_left_mask, c_right_mask], dim=-1)
                c_ones_mask = torch.ones(content_len, content_len, device=device)
                c_ones_mask[:a, :a] = c_all_mask
                c_ones_mask[a:, :a] = 0.0
                content_mask.append(c_ones_mask.unsqueeze(0))
                
                q_left_mask = torch.zeros((b , a - 2 * b), device=device)
                mask_triu = torch.triu(torch.ones((b, b), device=device), 0)
                q_mask = torch.cat([q_left_mask, 1 - mask_triu, mask_triu], dim=-1)
                q_ones_mask = torch.ones((query_len, content_len), device=device)
                q_ones_mask[:b, :a] = q_mask
                q_ones_mask[b:, :a] = 0.0
                query_mask.append(q_ones_mask.unsqueeze(0))
                temp_cache[key] = (content_mask[-1], query_mask[-1])
        query_mask = torch.cat(query_mask, dim=0)
        content_mask = torch.cat(content_mask, dim=0)
        return content_mask, query_mask

    def forward_nar_embedding(self, content_w, content_p, query_w, query_p):
        # embed positions
        positions = (
            self.embed_positions(content_w)
            if self.embed_positions is not None
            else None
        )
        if content_p is not None:
            reorder_content_positions = []
            for bsz_idx in range(content_p.size(0)):
                pos_item = positions[bsz_idx].index_select(0, content_p[bsz_idx])
                reorder_content_positions.append(pos_item.unsqueeze(0))
            reorder_content_positions = torch.cat(reorder_content_positions, dim=0)
            if query_p is not None:
                reorder_query_positions = []
                for bsz_idx in range(query_p.size(0)):
                    pos_item = positions[bsz_idx].index_select(0, query_p[bsz_idx])
                    reorder_query_positions.append(pos_item.unsqueeze(0))
                reorder_query_positions = torch.cat(reorder_query_positions, dim=0)    
        else:
            reorder_content_positions = positions
        
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(content_w)
        if reorder_content_positions is not None:
            x += reorder_content_positions
        x = self.dropout_module(x)
        decoder_padding_mask = content_w.eq(self.padding_idx)
        
        if query_w is not None:
            query_x = self.embed_scale * self.embed_tokens(query_w)
            if reorder_query_positions is not None:
                query_x += reorder_query_positions 
            query_x = self.dropout_module(query_x)
            query_padding_mask = query_w.eq(self.padding_idx)
        else:
            query_x = None
            query_padding_mask = None
        return x, decoder_padding_mask, query_x, query_padding_mask

    def forward_length_prediction(self, length_out, encoder_out, tgt_tokens=None):
        enc_feats = encoder_out["encoder_out"][0]  # T x B x C
        if len(encoder_out["encoder_padding_mask"]) > 0:
            src_masks = encoder_out["encoder_padding_mask"][0]  # B x T
        else:
            src_masks = None

        if tgt_tokens is not None:
            # obtain the length target
            tgt_lengs = tgt_tokens.ne(self.padding_idx).sum(1).long()
            length_tgt = tgt_lengs
            length_tgt = length_tgt.clamp(min=0, max=255)

        else:
            # predict the length target (greedy for now)
            # TODO: implementing length-beam
            pred_lengs = length_out.max(-1)[1]
            length_tgt = pred_lengs

        return length_tgt


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("janus_bart", "janus_bart_large")
def janus_bart_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

@register_model_architecture("janus_bart", "janus_bart_base")
def janus_bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    janus_bart_large_architecture(args)
