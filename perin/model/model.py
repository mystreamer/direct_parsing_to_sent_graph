#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import torch
import torch.nn as nn

from model.module.encoder import Encoder

from model.module.transformer import Decoder
from model.head.norec_head import NorecHead
from model.module.module_wrapper import ModuleWrapper
from utility.utils import create_padding_mask
from data.batch import Batch


class Model(nn.Module):
    def __init__(self, dataset, args, initialize=True):
        super(Model, self).__init__()
        self.encoder = Encoder(args, dataset)
        self.decoder = Decoder(args)

        self.heads = nn.ModuleList([])
        for i in range(len(dataset.child_datasets)):
            f, l = dataset.id_to_framework[i]
            self.heads.append(NorecHead(dataset.child_datasets[(f, l)], args, f, l, initialize))

        self.query_length = args.query_length
        self.label_smoothing = args.label_smoothing
        self.total_epochs = args.epochs
        self.dataset = dataset
        self.args = args

        self.share_weights()

    def forward(self, batch, inference=False, **kwargs):
        every_input, word_lens = batch["every_input"]
        decoder_lens = self.query_length * word_lens
        batch_size, input_len = every_input.size(0), every_input.size(1)
        device = every_input.device
        framework = batch["framework"][0].cpu().item()

        encoder_mask = create_padding_mask(batch_size, input_len, word_lens, device)
        decoder_mask = create_padding_mask(batch_size, self.query_length * input_len, decoder_lens, device)

        encoder_output, decoder_input = self.encoder(
            batch["input"], batch["char_form_input"], batch["input_scatter"], input_len, batch["framework"]
        )

        decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask, encoder_mask)
        decoder_output.requires_grad_(True)

        if inference:
            return {self.dataset.id_to_framework[framework]: self.heads[framework].predict(encoder_output, decoder_output, encoder_mask, decoder_mask, batch)}

        total_loss, losses, stats = 0.0, [], {}
        for i, head in enumerate(self.heads):
            if i == framework:
                total_loss_, losses_, stats_ = self.heads[framework](encoder_output, decoder_output, encoder_mask, decoder_mask, batch)

                total_loss = total_loss + total_loss_ / self.args.accumulation_steps
                losses.append(losses_)
                stats.update(stats_)
                continue

            args = self.get_dummy_batch(head, device)
            total_loss_, _, _ = head(*args)
            total_loss = total_loss + 0.0 * total_loss_
            losses.append([])

        return total_loss, losses, stats

    def get_params_for_optimizer(self, args):
        encoder_decay, encoder_no_decay = self.get_encoder_parameters()
        decoder_decay, decoder_no_decay = self.get_decoder_parameters()

        parameters = [
            {"params": encoder_decay, "weight_decay": args.encoder_weight_decay},
            {"params": encoder_no_decay, "weight_decay": 0.0},
            {"params": decoder_decay, "weight_decay": args.decoder_weight_decay},
            {"params": decoder_no_decay, "weight_decay": 0.0},
        ]
        return parameters

    def get_decoder_parameters(self):
        no_decay = ["bias", "LayerNorm.weight", "_norm.weight"]
        decay_params = (p for name, p in self.named_parameters() if not any(nd in name for nd in no_decay) and not name.startswith("encoder.bert") and "loss_weights" not in name and p.requires_grad)
        no_decay_params = (p for name, p in self.named_parameters() if any(nd in name for nd in no_decay) and not name.startswith("encoder.bert") and "loss_weights" not in name and p.requires_grad)

        return decay_params, no_decay_params

    def get_encoder_parameters(self):
        no_decay = ["bias", "LayerNorm.weight", "_norm.weight"]
        decay_params = (p for name, p in self.named_parameters() if not any(nd in name for nd in no_decay) and name.startswith(f"encoder.bert.encoder") and p.requires_grad)
        no_decay_params = (p for name, p in self.named_parameters() if any(nd in name for nd in no_decay) and name.startswith(f"encoder.bert.encoder") and p.requires_grad)

        return decay_params, no_decay_params

    def share_weights(self):
        ucca_heads = [head for i, head in enumerate(self.heads) if self.dataset.id_to_framework[i][0] == "ucca"]
        if len(ucca_heads) == 2:
            self.share_weights_(ucca_heads[0], ucca_heads[1], share_labels=True, share_edges=True, share_anchors=True)

        ptg_heads = [head for i, head in enumerate(self.heads) if self.dataset.id_to_framework[i][0] == "ptg"]
        if len(ptg_heads) == 2:
            self.share_weights_(ptg_heads[0], ptg_heads[1], share_edges=True, share_anchors=True)

        drg_heads = [head for i, head in enumerate(self.heads) if self.dataset.id_to_framework[i][0] == "drg"]
        if len(drg_heads) == 2:
            self.share_weights_(drg_heads[0], drg_heads[1], share_edges=True, share_tops=True, share_properties=True)

        amr_heads = [head for i, head in enumerate(self.heads) if self.dataset.id_to_framework[i][0] == "amr"]
        if len(amr_heads) == 2:
            self.share_weights_(amr_heads[0], amr_heads[1], share_edges=True, share_tops=True, share_properties=True)

    def share_weights_(self, a, b, share_edges=False, share_anchors=False, share_labels=False, share_tops=False, share_properties=False):
        if share_edges:
            del b.edge_classifier
            b.edge_classifier = ModuleWrapper(a.edge_classifier)

        if share_anchors:
            del b.anchor_classifier
            b.anchor_classifier = ModuleWrapper(a.anchor_classifier)

        if share_labels:
            del b.label_classifier
            b.label_classifier = ModuleWrapper(a.label_classifier)

        if share_properties:
            del b.property_classifier
            b.property_classifier = ModuleWrapper(a.property_classifier)

    def get_dummy_batch(self, head, device):
        encoder_output = torch.zeros(1, 1, self.args.hidden_size, device=device)
        decoder_output = torch.zeros(1, self.query_length, self.args.hidden_size, device=device)
        encoder_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
        decoder_mask = torch.zeros(1, self.query_length, dtype=torch.bool, device=device)
        batch = {
            "every_input": (torch.zeros(1, 1, dtype=torch.long, device=device), torch.ones(1, dtype=torch.long, device=device)),
            "input": (torch.zeros(1, 1, dtype=torch.long, device=device), torch.ones(1, dtype=torch.long, device=device)),
            "labels": (torch.zeros(1, 1, dtype=torch.long, device=device), torch.ones(1, dtype=torch.long, device=device)),
            "properties": torch.zeros(1, 1, 10, dtype=torch.long, device=device),
            "edge_presence": torch.zeros(1, 1, 1, dtype=torch.long, device=device),
            "edge_labels": (torch.zeros(1, 1, 1, head.dataset.edge_label_freqs.size(0), dtype=torch.long, device=device), torch.zeros(1, 1, 1, dtype=torch.bool, device=device)),
            "anchor": (torch.zeros(1, 1, 1, dtype=torch.long, device=device), torch.zeros(1, 1, dtype=torch.bool, device=device))
        }

        return encoder_output, decoder_output, encoder_mask, decoder_mask, batch
