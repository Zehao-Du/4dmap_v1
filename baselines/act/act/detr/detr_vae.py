# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from act.detr.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, action_dim, num_queries,
                 map4d_dim=0, map4d_token_dim=0, map4d_max_tokens=0):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        self.map4d_dim = map4d_dim
        self.map4d_token_dim = map4d_token_dim
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_state_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))

        if self.map4d_dim > 0:
            self.input_proj_map4d = nn.Sequential(
                nn.LayerNorm(self.map4d_dim),
                nn.Linear(self.map4d_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.input_proj_map4d = None

        # map4d as tokens: project each (object, timestep) to a token
        if self.map4d_token_dim > 0 and map4d_max_tokens > 0:
            self.map4d_token_proj = nn.Sequential(
                nn.Linear(map4d_token_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.map4d_token_pos = nn.Embedding(map4d_max_tokens, hidden_dim)
        else:
            self.map4d_token_proj = None

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        num_additional_tokens = 3 if self.input_proj_map4d is not None else 2
        self.additional_pos_embed = nn.Embedding(num_additional_tokens, hidden_dim) # latent, proprio, optional map4d

    def forward(self, obs, actions=None):
        is_training = actions is not None
        state = obs['state'] if self.backbones is not None else obs
        bs = state.shape[0]
        map4d_input = None
        if self.input_proj_map4d is not None:
            map4d_feature = obs.get('map4d_feature')
            if map4d_feature is None:
                raise ValueError("map4d_feature is required when DETRVAE is initialized with map4d_dim > 0")
            map4d_input = self.input_proj_map4d(map4d_feature)

        if is_training:
            # project CLS token, state sequence, and action sequence to embedding dim
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            state_embed = self.encoder_state_proj(state) # (bs, hidden_dim)
            state_embed = torch.unsqueeze(state_embed, axis=1)  # (bs, 1, hidden_dim)
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            # concat them together to form an input to the CVAE encoder
            encoder_input = torch.cat([cls_embed, state_embed, action_embed], axis=1) # (bs, seq+2, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+2, bs, hidden_dim)
            # no masking is applied to all parts of the CVAE encoder input
            is_pad = torch.full((bs, encoder_input.shape[0]), False).to(state.device) # False: not a padding
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
            # query CVAE encoder
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(state.device)
            latent_input = self.latent_out_proj(latent_sample)

        # CVAE decoder
        if self.backbones is not None:
            vis_data = obs['rgb']
            if "depth" in obs:
                vis_data = torch.cat([vis_data, obs['depth']], dim=2)
            num_cams = vis_data.shape[1]

            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id in range(num_cams):
                features, pos = self.backbones[0](vis_data[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature # (batch, hidden_dim, H, W)
                pos = pos[0] # (1, hidden_dim, H, W)
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            # proprioception features (state)
            proprio_input = self.input_proj_robot_state(state)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3) # (batch, hidden_dim, 4, 8)
            pos = torch.cat(all_cam_pos, axis=3) # (batch, hidden_dim, 4, 8)

            # map4d as tokens: project and append to src
            map4d_tokens_seq = None
            map4d_tokens_pos = None
            if self.map4d_token_proj is not None and 'map4d_tokens' in obs:
                raw_tokens = obs['map4d_tokens']  # (batch, T*N, token_dim)
                map4d_tokens_seq = self.map4d_token_proj(raw_tokens)  # (batch, T*N, hidden_dim)
                num_tokens = map4d_tokens_seq.shape[1]
                token_idx = torch.arange(num_tokens, device=state.device)
                map4d_tokens_pos = self.map4d_token_pos(token_idx)  # (T*N, hidden_dim)
                map4d_tokens_pos = map4d_tokens_pos.unsqueeze(0).expand(bs, -1, -1)  # (batch, T*N, hidden_dim)

            hs, memory = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                map4d_input,
                self.additional_pos_embed.weight,
                map4d_tokens_seq=map4d_tokens_seq,
                map4d_tokens_pos=map4d_tokens_pos,
            )
            hs = hs[0]  # (batch, num_queries, hidden_dim)
        else:
            state = self.input_proj_robot_state(state)
            hs, memory = self.transformer(
                None,
                None,
                self.query_embed.weight,
                None,
                latent_input,
                state,
                map4d_input,
                self.additional_pos_embed.weight,
            )
            hs = hs[0]

        # memory shape: (seq_len, batch, hidden_dim)
        a_hat = self.action_head(hs)
        return a_hat, [mu, logvar], memory


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder
