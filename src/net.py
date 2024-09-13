from functools import partial
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block, Attention
from timm.models.layers import Mlp

from util.pos_embed import get_2d_sincos_pos_embed
from util.helpers import to_2tuple


################# Layers ########################
class Conv2dLayerPartial(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 mask_channels,  # Number of mask channels.
                 kernel_size,  # Width and height of the convolution kernel.
                 stride,  # Stride for the convolution kernel
                 ):
        super().__init__()
        self.slide_winsize = kernel_size ** 2
        self.stride = stride
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.weight_maskUpdater = torch.ones(out_channels, mask_channels, kernel_size, kernel_size)

    def forward(self, x, mask=None):
        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding)
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)  # 0 or 1
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            return x, None


class mlp_block(nn.Module):
    def __init__(self, dim1, dim2, activation='gelu', norm=True, dropout=0.0):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.activation_dict = {'sigmodi': nn.Sigmoid(), 'tanh': nn.Tanh(), 'softmax': nn.Softmax(dim=-1),
                                'relu': nn.ReLU(),
                                'elu': nn.ELU(), 'swish': nn.SiLU(), 'gelu': nn.GELU()}

        layers = [torch.nn.Linear(dim1, dim2)]

        if self.norm:
            layers.append(nn.BatchNorm1d(dim2))

        if self.activation:
            layers.append(self.activation_dict[self.activation])

        layers.append(nn.Dropout(self.dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class MaxoutMLP(nn.Module):
    def __init__(self, in_feature, out_feature, hiden_features=None, num_pieces=2, **kwargs):
        super(MaxoutMLP, self).__init__()
        hiden_features = hiden_features or (in_feature * 4)

        self.fc1 = nn.Linear(in_feature, hiden_features * num_pieces)
        self.fc2 = nn.Linear(hiden_features, out_feature)
        self.num_pieces = num_pieces

    def forward(self, x):
        x = F.max_pool1d(self.fc1(x), self.num_pieces).squeeze(1)
        x = self.fc2(x)
        return x


################# Blocks ########################


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, dropout=0.0, out_channels=None, ch=None, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        ch = in_channels if out_channels is None else ch
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.GroupNorm(num_groups=ch, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv2dLayerPartial(in_channels,
                                        out_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1)

        self.norm2 = torch.nn.GroupNorm(num_groups=ch, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = Conv2dLayerPartial(out_channels,
                                        out_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, mask):
        h = x
        h = self.norm1(h)
        h = F.silu(h)  # silu, gelu
        h, m = self.conv1(h, mask)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h, m = self.conv2(h, m)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h, m


################# Networks ########################

class Encoder_sat(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, ch_mult=(2, 2), dropout=0.0, temb_channels=2):
        super().__init__()

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.conv_in = Conv2dLayerPartial(in_channels * 2,
                                          ch,
                                          in_channels,
                                          kernel_size=3,
                                          stride=1)

        self.temb_channels = temb_channels
        if temb_channels > 0:
            self.temb_proj = Mlp(temb_channels, hidden_features=ch * 4, out_features=ch)
        else:
            self.temb_proj = None

        block_in = ch

        self.res_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             dropout=dropout, ch=ch))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=ch, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = Conv2dLayerPartial(block_in,
                                           out_channels,
                                           block_in,
                                           kernel_size=3,
                                           stride=1)

    def forward(self, x, mask):
        temb = torch.mean(x[:, :, :, :self.temb_channels], dim=(1, 2))
        h = torch.cat([x[:, :, :, self.temb_channels:], mask[:, :, :, self.temb_channels:]], dim=1).float()
        m = mask[:, :, :, self.temb_channels:].float()
        h, m = self.conv_in(h, m)
        if self.temb_proj is not None:
            h = h + self.temb_proj(temb).reshape(h.shape[0], -1, 1, 1)
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h, m = self.res_blocks[i_level][i_block](h, m)
        h = self.norm_out(h)
        h = F.silu(h)
        h, m = self.conv_out(h, m)
        return h, m


class Encoder_all(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, drop_path=0.0, act_layer=nn.GELU, out_chans=None):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans if out_chans else in_chans
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        assert self.num_patch[0] * self.num_patch[1] == num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        print(embed_dim, num_heads)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path, act_layer=act_layer) for i in
             range(depth)])  # , qk_scale=None
        self.norm = norm_layer(embed_dim)
        self.decoder_pred = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * self.out_chans, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patch, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        p0 = self.patch_size[0]
        p1 = self.patch_size[1]
        h = self.num_patch[0]
        w = self.num_patch[1]
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p0, p1, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p0, w * p1))
        return imgs

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)

        # predictor projection
        latent = self.decoder_pred(latent)

        output = self.unpatchify(latent[:, 1:])

        return output


################# Models ########################

class Fluex_net(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., rescale_loss=False, step_two=False, res=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.res = res
        self.drop_sat = drop_sat

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)
            # self.out_shell_net = Mlp(9, 4, 4, act_layer=act_layer)

            # self.out_shell_net = MaxoutMLP(9, 4, hiden_features=None, num_pieces=2)

            # self.out_shell_net = Encoder_all(img_size=img_size, patch_size=(patch_size[0], 3), in_chans=1,
            #                                  embed_dim=latent_dim, depth=1, num_heads=num_heads,
            #                                  mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)
            # self.conv_out2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        if not self.rescale_loss:
            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        else:
            # s2v_loss1 = torch.sum(torch.abs(output - label) * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()
            # s2v_loss2 = torch.sum((output - label) ** 2 * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()

            weight = torch.ones(27).reshape(1, 1, 1, -1).to(output.device)
            weight[:, :, :, 1:5] = 2

            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label) * weight) / 31 * 27
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label) * weight) / 31 * 27

        tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + tv_h * self.lambda_g3v + tv_w * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        h = self.encoder(h * m)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

            # if self.res:
            #     output2 = self.out_shell_net(output[:, :, :, :7].detach())
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7] + output[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:] + output[:, :, :, 17:]],
            #                        dim=-1)
            # else:
            #     output2 = self.out_shell_net(output)
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label)

        return loss, output


class Fluex_net_cnnonly(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., rescale_loss=False, step_two=False, res=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.res = res
        self.drop_sat = drop_sat

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)
            # self.out_shell_net = Mlp(9, 4, 4, act_layer=act_layer)

            # self.out_shell_net = MaxoutMLP(9, 4, hiden_features=None, num_pieces=2)

            # self.out_shell_net = Encoder_all(img_size=img_size, patch_size=(patch_size[0], 3), in_chans=1,
            #                                  embed_dim=latent_dim, depth=1, num_heads=num_heads,
            #                                  mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)
            # self.conv_out2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        if not self.rescale_loss:
            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        else:
            # s2v_loss1 = torch.sum(torch.abs(output - label) * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()
            # s2v_loss2 = torch.sum((output - label) ** 2 * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()

            weight = torch.ones(27).reshape(1, 1, 1, -1).to(output.device)
            weight[:, :, :, 1:5] = 2

            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label) * weight) / 31 * 27
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label) * weight) / 31 * 27

        tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + tv_h * self.lambda_g3v + tv_w * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

            # if self.res:
            #     output2 = self.out_shell_net(output[:, :, :, :7].detach())
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7] + output[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:] + output[:, :, :, 17:]],
            #                        dim=-1)
            # else:
            #     output2 = self.out_shell_net(output)
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label)

        return loss, output

class Fluex_net_small(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., rescale_loss=False, step_two=False, res=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.res = res
        self.drop_sat = drop_sat

        self.rescale_loss = rescale_loss

        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.shell_num = shell_num

        self.source_embedding = nn.Embedding(num_sat, in_chans * img_size[1])
        self.sat_blocks = Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = Encoder_all(img_size=img_size, patch_size=(patch_size[0], 3), in_chans=1,
                                             embed_dim=latent_dim, depth=1, num_heads=num_heads,
                                             mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)
            self.conv_out2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        if not self.rescale_loss:
            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        else:
            s2v_loss1 = torch.sum(torch.abs(output - label) * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()
            s2v_loss2 = torch.sum((output - label) ** 2 * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()

        tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + tv_h * self.lambda_g3v + tv_w * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []
        source_ids = torch.arange(self.num_sat).to(inputs.device)

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat, device=inputs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i in range(self.num_sat):
            if i in ids_keep or (not self.training):
                source_embeds = self.source_embedding(source_ids[i]).reshape(1, self.in_chans, 1, self.img_size[1]).expand(inputs.shape[0], -1,
                                                                                                                           self.img_size[0], -1)
                source_input = torch.cat([inputs[:, i, :, :, :self.shell_num], inputs[:, i, :, :, self.shell_num:] + source_embeds], dim=-1)
                output, update_mask = self.sat_blocks(source_input, mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        h = self.encoder(h * m)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            if self.res:
                output2 = self.out_shell_net(output.detach())
                output2 = self.conv_out2(output2)
                output = torch.cat([output2[:, :, :, :7] + output[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:] + output[:, :, :, 17:]],
                                   dim=-1)
            else:
                output2 = self.out_shell_net(output)
                output2 = self.conv_out2(output2)
                output = torch.cat([output2[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label)

        return loss, output


class Fluex_net_single(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., rescale_loss=False, step_two=False, res=True, target_shell=0):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.res = res
        self.target_shell = target_shell
        self.drop_sat = drop_sat

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_c = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv1d(img_size[1], 1, kernel_size=3, stride=1, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        if not self.rescale_loss:
            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        else:
            s2v_loss1 = torch.sum(torch.abs(output - label) * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()
            s2v_loss2 = torch.sum((output - label) ** 2 * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()

        tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + tv_h * self.lambda_g3v + tv_w * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat, device=inputs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        h = self.encoder(h * m)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_c(h).squeeze(1).permute(0, 2, 1)
        output = self.conv_l(output).reshape(-1, 1, output.shape[2], 1)

        loss = self.forward_loss(label, output, mask_label)

        return loss, output


class Fluex_net_mlp(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., rescale_loss=False, step_two=False, res=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.res = res
        self.drop_sat = drop_sat

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        dim = [512, 256, 128, 64, 32]
        flatten_dim = img_size[0] * img_size[1] * in_chans

        layers = [mlp_block(flatten_dim * 2, dim[0], 'gelu', norm=True, dropout=drop_path)]

        for i in range(1, len(dim)):
            layers.append(mlp_block(dim[i - 1], dim[i], 'gelu', norm=True, dropout=drop_path))

        self.sat_blocks = nn.Sequential(*layers)

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.shell_num = shell_num

        self.source_embedding = nn.Embedding(num_sat, in_chans * img_size[1])

        dim2 = [32, 64, 128, 256, 512]
        layers = [mlp_block(dim[-1], dim2[0], 'gelu', norm=True, dropout=drop_path)]

        for i in range(1, len(dim2)):
            layers.append(mlp_block(dim2[i - 1], dim2[i], 'gelu', norm=True, dropout=drop_path))
        layers.append(mlp_block(dim2[-1], img_size[0] * img_size[1], 'gelu', norm=True, dropout=drop_path))

        self.encoder = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        if not self.rescale_loss:
            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        else:
            s2v_loss1 = torch.sum(torch.abs(output - label) * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()
            s2v_loss2 = torch.sum((output - label) ** 2 * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()

        tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + tv_h * self.lambda_g3v + tv_w * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h = []
        source_ids = torch.arange(self.num_sat).to(inputs.device)

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i in range(self.num_sat):
            if i in ids_keep or (not self.training):
                source_embeds = self.source_embedding(source_ids[i]).reshape(1, self.in_chans, 1, self.img_size[1]).expand(inputs.shape[0], -1,
                                                                                                                           self.img_size[0], -1)
                source_input = torch.cat([inputs[:, i, :, :, :self.shell_num], inputs[:, i, :, :, self.shell_num:] + source_embeds], dim=-1)
                sat_input = torch.cat([source_input.flatten(1, -1), mask[:, i].flatten(1, -1)], dim=1)
                output = self.sat_blocks(sat_input)
                h.append(output)
        h = torch.mean(torch.stack(h), dim=0)

        output = self.encoder(h).view(-1, label.shape[1], label.shape[2], label.shape[3])

        loss = self.forward_loss(label, output, mask_label)

        return loss, output


class Fluex_net_omni(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., rescale_loss=False, step_two=False, res=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.shell_num = shell_num
        self.res = res
        self.drop_sat = drop_sat

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans - 1
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.omni_chans = 8

        self.omni_net = Encoder_all(img_size=(img_size[0], 1), patch_size=(9, 1), in_chans=1, out_chans=self.omni_chans,
                                    embed_dim=self.omni_chans, depth=2, num_heads=2,
                                    mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim + self.omni_chans, out_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        if not self.rescale_loss:
            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        else:
            # s2v_loss1 = torch.sum(torch.abs(output - label) * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()
            # s2v_loss2 = torch.sum((output - label) ** 2 * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()

            weight = torch.ones(27).reshape(1, 1, 1, -1).to(output.device)
            weight[:, :, :, 1:5] = 2

            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label) * weight) / 31 * 27
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label) * weight) / 31 * 27

        tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + tv_h * self.lambda_g3v + tv_w * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        omni = inputs[:, :, -1, :, self.shell_num:]  # B x S x T x W
        # omni_mask = mask[:, :, -1, :, self.shell_num:]
        # inputs = inputs[:, :, :-1, :, :]
        # mask = mask[:, :, :-1, :, :]

        omni = omni.mean(dim=(1, 3)).view(omni.shape[0], 1, omni.shape[2], 1).detach()  # B x 1 x T x 1
        omni = self.omni_net(omni)

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        h = torch.cat([h * m, omni.repeat((1, 1, 1, h.shape[-1]))], dim=1)

        h = self.encoder(h)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label)

        return loss, output


class Fluex_net_avg(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., rescale_loss=False, step_two=False, res=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.res = res
        self.drop_sat = drop_sat

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)

            # self.out_shell_net = MaxoutMLP(9, 4, hiden_features=None, num_pieces=2)

            # self.out_shell_net = Encoder_all(img_size=img_size, patch_size=(patch_size[0], 3), in_chans=1,
            #                                  embed_dim=latent_dim, depth=1, num_heads=num_heads,
            #                                  mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)
            # self.conv_out2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        if not self.rescale_loss:
            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        else:
            # s2v_loss1 = torch.sum(torch.abs(output - label) * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()
            # s2v_loss2 = torch.sum((output - label) ** 2 * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()

            weight = torch.ones(27).reshape(1, 1, 1, -1).to(output.device)
            weight[:, :, :, 1:5] = 2

            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label) * weight) / 31 * 27
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label) * weight) / 31 * 27

        tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + tv_h * self.lambda_g3v + tv_w * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        h = self.encoder(h * m)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

            # if self.res:
            #     output2 = self.out_shell_net(output[:, :, :, :7].detach())
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7] + output[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:] + output[:, :, :, 17:]],
            #                        dim=-1)
            # else:
            #     output2 = self.out_shell_net(output)
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:]], dim=-1)

        mean_hrs = 3
        B, C, tlong, W = label.shape
        new_T = (tlong // mean_hrs) * mean_hrs

        pooled_output = output[:, :, :new_T].view(B, C, -1, mean_hrs, W)
        pooled_output = torch.mean(pooled_output, dim=3)

        tem_tensor = label * (1 - mask_label)
        tem_tensor2 = tem_tensor[:, :, :new_T].view(B, C, -1, mean_hrs, W)

        pooled_mask_label = 1 - mask_label
        pooled_mask_label = pooled_mask_label.view(B, C, -1, mean_hrs, W).float()

        sums = (tem_tensor2 * pooled_mask_label).sum(dim=3)
        counts = pooled_mask_label.sum(dim=3)
        pooled_label = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))

        pooled_mask_label = (counts == 0).float()

        loss = self.forward_loss(pooled_label, pooled_output, pooled_mask_label)

        return loss, output


class Fluex_net_argument(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., rescale_loss=False, step_two=False, res=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.res = res
        self.drop_sat = drop_sat
        self.shell_num = shell_num

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(7, 4)

            # self.out_shell_net = MaxoutMLP(9, 4, hiden_features=None, num_pieces=2)

            # self.out_shell_net = Encoder_all(img_size=img_size, patch_size=(patch_size[0], 3), in_chans=1,
            #                                  embed_dim=latent_dim, depth=1, num_heads=num_heads,
            #                                  mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)
            # self.conv_out2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        if not self.rescale_loss:
            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        else:
            # s2v_loss1 = torch.sum(torch.abs(output - label) * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()
            # s2v_loss2 = torch.sum((output - label) ** 2 * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()

            weight = torch.ones(27).reshape(1, 1, 1, -1).to(output.device)
            weight[:, :, :, 1:5] = 2

            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label) * weight) / 31 * 27
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label) * weight) / 31 * 27

        tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + tv_h * self.lambda_g3v + tv_w * self.lambda_g4v

        return loss

    def shell_flip(self, inputs, l_ind=18, label=False):
        if label:
            inputs_middle = inputs[:, :, :, :l_ind]
            inputs_after = inputs[:, :, :, l_ind:]

            inputs_middle = inputs_middle.flip(dims=(-1,))

            inputs = torch.cat((inputs_middle, inputs_after), dim=-1)
        else:
            inputs_before = inputs[:, :, :, :, :self.shell_num]
            inputs_middle = inputs[:, :, :, :, self.shell_num:l_ind + self.shell_num]
            inputs_after = inputs[:, :, :, :, self.shell_num + l_ind:]

            inputs_middle = inputs_middle.flip(dims=(-1,))

            inputs = torch.cat((inputs_before, inputs_middle, inputs_after), dim=-1)
        return inputs

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        p = 0.3
        if self.training and torch.rand(1).item() < p:
            inputs = self.shell_flip(inputs)
            label = self.shell_flip(label, label=True)
            mask = self.shell_flip(mask)
            mask_label = self.shell_flip(mask_label, label=True)
        # if self.training and torch.rand(1).item() < p:
        #     inputs = inputs.flip(dims=(-2,))
        #     label = label.flip(dims=(-2,))
        #     mask = mask.flip(dims=(-2,))
        #     mask_label = mask_label.flip(dims=(-2,))

        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        h = self.encoder(h * m)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:12].detach().reshape(B, C * T, 7)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

            # if self.res:
            #     output2 = self.out_shell_net(output[:, :, :, :7].detach())
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7] + output[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:] + output[:, :, :, 17:]],
            #                        dim=-1)
            # else:
            #     output2 = self.out_shell_net(output)
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label)

        return loss, output


class Fluex_net_label(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., rescale_loss=False, step_two=False, res=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.res = res
        self.drop_sat = drop_sat
        self.shell_num = shell_num

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(
                Encoder_sat(in_chans + 1, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            # self.out_shell_net = nn.Linear(9, 4)
            self.out_shell_net = Mlp(9, 4, 4, act_layer=act_layer)

            # self.out_shell_net = MaxoutMLP(9, 4, hiden_features=None, num_pieces=2)

            # self.out_shell_net = Encoder_all(img_size=img_size, patch_size=(patch_size[0], 3), in_chans=1,
            #                                  embed_dim=latent_dim, depth=1, num_heads=num_heads,
            #                                  mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)
            # self.conv_out2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        if not self.rescale_loss:
            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        else:
            # s2v_loss1 = torch.sum(torch.abs(output - label) * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()
            # s2v_loss2 = torch.sum((output - label) ** 2 * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()

            weight = torch.ones(27).reshape(1, 1, 1, -1).to(output.device)
            weight[:, :, :, 1:5] = 2

            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label) * weight) / 31 * 27
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label) * weight) / 31 * 27

        tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + tv_h * self.lambda_g3v + tv_w * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        t_tensor = inputs[:, 0, :1, :, :self.shell_num]
        input_label = torch.cat([t_tensor.detach().clone(), label.detach().clone()], dim=-1)
        input_label[:, :, :, :7] = 0
        input_label_mask = torch.cat([t_tensor.detach().clone() * 0 + 1, 1 - mask_label.detach().clone()], dim=-1)
        input_label_mask[:, :, :, :7] = 0

        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                il = torch.cat([inputs[:, i], input_label], dim=1)
                ilm = torch.cat([mask[:, i], input_label_mask], dim=1)
                output, update_mask = encoder(il, ilm)
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        h = self.encoder(h * m)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

            # if self.res:
            #     output2 = self.out_shell_net(output[:, :, :, :7].detach())
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7] + output[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:] + output[:, :, :, 17:]],
            #                        dim=-1)
            # else:
            #     output2 = self.out_shell_net(output)
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label)

        return loss, output


class Fluex_net_gradient(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., rescale_loss=False, step_two=False, res=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.res = res
        self.drop_sat = drop_sat
        self.shell_num = shell_num

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)
            # self.out_shell_net = Mlp(9, 4, 4, act_layer=act_layer)

            # self.out_shell_net = MaxoutMLP(9, 4, hiden_features=None, num_pieces=2)

            # self.out_shell_net = Encoder_all(img_size=img_size, patch_size=(patch_size[0], 3), in_chans=1,
            #                                  embed_dim=latent_dim, depth=1, num_heads=num_heads,
            #                                  mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)
            # self.conv_out2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        if not self.rescale_loss:
            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        else:
            # s2v_loss1 = torch.sum(torch.abs(output - label) * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()
            # s2v_loss2 = torch.sum((output - label) ** 2 * (1 - mask_label)) / torch.sum((1 - mask_label)).detach()

            weight = torch.ones(27).reshape(1, 1, 1, -1).to(output.device)
            weight[:, :, :, 1:5] = 2

            s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label) * weight) / 31 * 27
            s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label) * weight) / 31 * 27

        tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + tv_h * self.lambda_g3v + tv_w * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                # il = torch.cat([inputs[:, i], input_label], dim=1)
                # ilm = torch.cat([mask[:, i], input_label_mask], dim=1)
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        h = self.encoder(h * m)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

            # if self.res:
            #     output2 = self.out_shell_net(output[:, :, :, :7].detach())
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7] + output[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:] + output[:, :, :, 17:]],
            #                        dim=-1)
            # else:
            #     output2 = self.out_shell_net(output)
            #     output2 = self.conv_out2(output2)
            #     output = torch.cat([output2[:, :, :, :7], output[:, :, :, 7:17], output2[:, :, :, 17:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label)

        grad_tensor = torch.diff(label, dim=-1)
        grad_tensor = F.pad(grad_tensor, (0, 1))
        grad_mask = (1 - mask_label)[..., :-1] * (1 - mask_label)[..., 1:]
        grad_mask = F.pad(grad_mask, (0, 1))

        grad_output = torch.diff(output, dim=-1)
        grad_output = F.pad(grad_output, (0, 1))

        g_loss = self.forward_loss(grad_tensor, grad_output, grad_mask)

        loss = loss + g_loss

        return loss, output


class Fluex_net_meta_old(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., meta_dim=1, rescale_loss=False, step_two=False, res=True, means=4.04,
                 stds=1.14):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.shell_num = shell_num
        self.res = res
        self.drop_sat = drop_sat
        self.meta_dim = meta_dim

        self.p = (np.log10(1500) - means) / stds
        self.q = (np.log10(4500) - means) / stds
        self.p, self.q = self.p.item(), self.q.item()

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans - 1
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.omni_chans = 8

        self.meta_net = Encoder_sat(meta_dim, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim, out_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def float_to_index_and_weights(self, float_mask):
        indices = ((float_mask - 2.8) / 0.2).floor().long()
        fractions = ((float_mask - 2.8) / 0.2) % 1

        weights_lower = 1 - fractions
        weights_upper = fractions

        # Handle edge cases (less than 2.8 and greater than 8.0)
        less_mask = float_mask < 2.8
        more_mask = float_mask > 8.0

        indices[less_mask] = 0
        indices[more_mask] = 27  # this is because 8.0 corresponds to the 27th index (0-indexed)

        weights_lower[less_mask | more_mask] = 1.0
        weights_upper[less_mask | more_mask] = 0.0

        return indices, weights_lower, weights_upper

    def meta_loss(self, output, float_mask):
        indices, weights_lower, weights_upper = self.float_to_index_and_weights(float_mask)

        # Use gather for lower and upper indices
        selected_values_lower = output.gather(3, indices.unsqueeze(1).unsqueeze(3)).squeeze(3)
        selected_values_upper = output.gather(3, (indices + 1).unsqueeze(1).unsqueeze(3)).squeeze(3)

        combined_values = weights_lower * selected_values_lower + weights_upper * selected_values_upper

        check_mask = (float_mask > 0).float()

        # Compute losses for values below and above the range
        lower_loss = F.relu(self.p - combined_values) * check_mask
        upper_loss = F.relu(combined_values - self.q) * check_mask

        less_mask = (float_mask < 2.8).unsqueeze(1).float()
        less_than_q_loss = F.relu(self.q - combined_values) * less_mask
        lower_loss *= (1 - less_mask)
        upper_loss *= (1 - less_mask)

        # Combine the losses
        total_loss = (lower_loss + upper_loss + less_than_q_loss).mean()

        return total_loss

    def monotonicity_loss(self, output):
        # Assuming output is of shape [batch_size, ..., sequence_length]
        diffs = output[..., 1:] - output[..., :-1]
        penalty = torch.mean(F.relu(-diffs))  # Penalize negative differences
        return penalty

    def forward_loss(self, label, output, mask_label, meta_data, meta_data_mask):
        s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
        s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))
        if not self.rescale_loss:
            meta_loss = 0
        else:
            float_mask = meta_data * (1 - meta_data_mask)
            meta_loss = self.meta_loss(output, float_mask)

        # tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        # tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        monotonicity_loss = self.monotonicity_loss(output)

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + meta_loss + monotonicity_loss * self.lambda_g3v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        common_part = meta_data[:, :, :, :2].expand(-1, self.meta_dim, -1, -1)
        channels = meta_data[:, :, :, 2:].transpose(1, 3)
        meta = torch.cat([common_part, channels], dim=-1)
        common_part = meta_data_mask[:, :, :, :2].expand(-1, self.meta_dim, -1, -1)
        channels = meta_data_mask[:, :, :, 2:].transpose(1, 3)
        meta_mask = torch.cat([common_part, channels], dim=-1)
        meta, meta_mask = self.meta_net(meta, meta_mask)

        h = h * m + meta * meta_mask

        h = self.encoder(h)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label, meta_data[:, 0, :, -1], meta_data_mask[:, 0, :, -1])

        return loss, output


class Fluex_net_meta(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., meta_dim=1, rescale_loss=False, step_two=False, res=True, means=4.05,
                 stds=1.14, meta_feature=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.shell_num = shell_num
        self.res = res
        self.drop_sat = drop_sat
        self.meta_dim = meta_dim
        self.meta_feature = meta_feature
        self.means = means
        self.stds = stds

        self.p = (np.log10(1500) - means) / stds
        self.q = (np.log10(4500) - means) / stds
        self.m = (np.log10(3000) - means) / stds
        self.p, self.q, self.m = self.p.item(), self.q.item(), self.m.item()
        self.range = self.q - self.p

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans - 1
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.omni_chans = 8

        self.meta_net, self.meta_net2 = None, None
        if meta_feature:
            self.meta_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)
            if meta_dim > 1:
                self.meta_net2 = Encoder_sat(meta_dim-1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim, out_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def float_to_index_and_weights(self, float_mask):
        indices = ((float_mask - 2.8) / 0.2)
        fractions = ((float_mask - 2.8) / 0.2) % 1

        weights_lower = 1 - fractions
        weights_upper = fractions

        # Handle edge cases (less than 2.8 and greater than 8.0)
        less_mask = float_mask < 2.8
        more_mask = float_mask > 8.0

        indices[less_mask] = 0
        indices[more_mask] = 26  # this is because 8.0 corresponds to the 27th index (0-indexed)

        weights_lower[less_mask | more_mask] = 1.0
        weights_upper[less_mask | more_mask] = 0.0

        return indices, weights_lower, weights_upper

    def calculate_fx_and_indicator_map(self, meta_data, meta_data_mask):
        meta_data = meta_data[:, 0, :, -1]
        meta_data_mask = meta_data_mask[:, 0, :, -1]
        float_mask = meta_data * (1 - meta_data_mask)
        indices, weights_lower, weights_upper = self.float_to_index_and_weights(float_mask)

        x0_tensor = indices.unsqueeze(-1)
        B, T, _ = x0_tensor.shape

        # Generate x values from 0 to 25 (27 elements)
        x_values = torch.arange(0, 27, dtype=x0_tensor.dtype, device=x0_tensor.device)

        # Calculate f(x) for each x in x_values
        # fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)
        fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)

        # Calculate floor and ceil for each x0
        floor_x = torch.floor(x0_tensor).to(torch.int64)
        ceil_x = torch.ceil(x0_tensor).to(torch.int64)

        # Create indicator map (initialize with False)
        indicator_map = torch.zeros(B, T, 27, dtype=torch.bool, device=x0_tensor.device)

        # Set True at positions corresponding to floor(x) and ceil(x)
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        equal_mask = floor_x == ceil_x
        floor_x[equal_mask] = torch.clamp(floor_x[equal_mask] - 1, min=0)
        ceil_x[equal_mask] = torch.clamp(ceil_x[equal_mask] + 1, max=26)
        # Update the indicator_map with the adjusted values
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        fx = fx * indicator_map.float()

        return fx.unsqueeze(1), indicator_map.unsqueeze(1), indices, weights_lower, weights_upper

    def monotonicity_loss(self, output):
        # Assuming output is of shape [batch_size, ..., sequence_length]
        diffs = output[..., 1:] - output[..., :-1]
        penalty = torch.mean(F.relu(-diffs))  # Penalize negative differences
        return penalty

    def forward_loss(self, label, output, mask_label, fx, indicator_map):
        s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
        s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))

        meta_loss1 = torch.abs(output - fx) * indicator_map * mask_label
        meta_loss2 = (output - fx) ** 2 * indicator_map * mask_label

        if self.rescale_loss:
            meta_loss_mask = (meta_loss1 > self.range).float()
            meta_loss1 = meta_loss1 * meta_loss_mask
            meta_loss2 = meta_loss2 * meta_loss_mask

        meta_loss = torch.mean(meta_loss1 * self.lambda_g1v + meta_loss2 * self.lambda_g2v)

        # tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        # tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        monotonicity_loss = self.monotonicity_loss(output)

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + monotonicity_loss * self.lambda_g3v + meta_loss * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        if self.drop_sat >= 0:
            len_keep = int(self.num_sat * (1 - self.drop_sat))
        elif (not self.training) and self.drop_sat < 0:
            len_keep = int(self.num_sat)
        else:
            len_keep = random.randint(int(abs(self.drop_sat)), int(self.num_sat))
        noise = torch.rand(self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_keep = ids_shuffle[:len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep:  #  or (not self.training)
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        fx, indicator_map, indices, weights_lower, weights_upper = self.calculate_fx_and_indicator_map(meta_data.detach(), meta_data_mask.detach())

        if self.meta_feature:
            common_part = meta_data[:, :, :, :2].expand(-1, 1, -1, -1)
            time_fx = torch.cat([common_part, fx], dim=-1)
            common_part = meta_data_mask[:, :, :, :2].expand(-1, 1, -1, -1)
            time_indicator_map = torch.cat([common_part, indicator_map], dim=-1)
            meta, meta_mask = self.meta_net(time_fx, time_indicator_map)
            h = h * m + meta * meta_mask

            if self.meta_dim > 1:
                common_part = meta_data[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data[:, :, :, 2:-1].transpose(1, 3)
                omni = torch.cat([common_part, channels], dim=-1)
                common_part = meta_data_mask[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data_mask[:, :, :, 2:-1].transpose(1, 3)
                omni_mask = torch.cat([common_part, channels], dim=-1)
                omni, omni_mask = self.meta_net2(omni, omni_mask)

                h = h + omni * omni_mask

        h = self.encoder(h)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label, fx, indicator_map)

        return loss, output


class Fluex_net_meta_future(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., meta_dim=1, rescale_loss=False, step_two=False, res=True, means=4.05,
                 stds=1.14, meta_feature=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.shell_num = shell_num
        self.res = res
        self.drop_sat = drop_sat
        self.meta_dim = meta_dim
        self.meta_feature = meta_feature
        self.means = means
        self.stds = stds

        self.p = (np.log10(1500) - means) / stds
        self.q = (np.log10(4500) - means) / stds
        self.m = (np.log10(3000) - means) / stds
        self.p, self.q, self.m = self.p.item(), self.q.item(), self.m.item()
        self.range = self.q - self.p

        self.rescale_loss = rescale_loss

        self.nowcast_model = Fluex_net_meta(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                           num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                           lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                           res=res, meta_dim=meta_dim, means=means, stds=stds, meta_feature=meta_feature)

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans - 1
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.omni_chans = 8

        self.meta_net, self.meta_net2 = None, None
        if meta_feature:
            self.meta_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)
            if meta_dim > 1:
                self.meta_net2 = Encoder_sat(meta_dim-1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)

        self.forecast_model = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=1, out_chans=1,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)
        self.conv_out = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_loss(self, label, output, mask_label):
        s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
        s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        _, inpainting = self.nowcast_model(inputs, label, mask, mask_label, meta_data, meta_data_mask)

        inpainting = self.forecast_model(inpainting)
        output = self.conv_out(inpainting)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label)

        return loss, output


class Fluex_net_meta2(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., meta_dim=1, rescale_loss=False, step_two=False, res=True, means=4.05,
                 stds=1.14, meta_feature=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.shell_num = shell_num
        self.res = res
        self.drop_sat = drop_sat
        self.meta_dim = meta_dim
        self.meta_feature = meta_feature
        self.means = means
        self.stds = stds

        self.p = (np.log10(1500) - means) / stds
        self.q = (np.log10(4500) - means) / stds
        self.m = (np.log10(3000) - means) / stds
        self.p, self.q, self.m = self.p.item(), self.q.item(), self.m.item()
        self.range = self.q - self.p

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans - 1
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.omni_chans = 8

        self.meta_net, self.meta_net2 = None, None
        if meta_feature:
            self.meta_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)
            if meta_dim > 1:
                self.meta_net2 = Encoder_sat(meta_dim-1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim, out_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def float_to_index_and_weights(self, float_mask):
        indices = ((float_mask - 2.8) / 0.2)
        fractions = ((float_mask - 2.8) / 0.2) % 1

        weights_lower = 1 - fractions
        weights_upper = fractions

        # Handle edge cases (less than 2.8 and greater than 8.0)
        less_mask = float_mask < 2.8
        more_mask = float_mask > 8.0

        indices[less_mask] = 0
        indices[more_mask] = 26  # this is because 8.0 corresponds to the 27th index (0-indexed)

        weights_lower[less_mask | more_mask] = 1.0
        weights_upper[less_mask | more_mask] = 0.0

        return indices, weights_lower, weights_upper

    def calculate_fx_and_indicator_map(self, meta_data, meta_data_mask):
        meta_data = meta_data[:, 0, :, -1]
        meta_data_mask = meta_data_mask[:, 0, :, -1]
        float_mask = meta_data * (1 - meta_data_mask)
        indices, weights_lower, weights_upper = self.float_to_index_and_weights(float_mask)

        x0_tensor = indices.unsqueeze(-1)
        B, T, _ = x0_tensor.shape

        # Generate x values from 0 to 25 (27 elements)
        x_values = torch.arange(0, 27, dtype=x0_tensor.dtype, device=x0_tensor.device)

        # Calculate f(x) for each x in x_values
        # fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)
        fx = 1500 * x_values.unsqueeze(0).unsqueeze(0) + (3000 - 1500 * x0_tensor)
        fx = torch.clip(fx, 1e-5)
        fx = (torch.log10(fx) - self.means) / self.stds


        # Calculate floor and ceil for each x0
        floor_x = torch.floor(x0_tensor).to(torch.int64)
        ceil_x = torch.ceil(x0_tensor).to(torch.int64)

        # Create indicator map (initialize with False)
        indicator_map = torch.zeros(B, T, 27, dtype=torch.bool, device=x0_tensor.device)

        # Set True at positions corresponding to floor(x) and ceil(x)
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        equal_mask = floor_x == ceil_x
        floor_x[equal_mask] = torch.clamp(floor_x[equal_mask] - 1, min=0)
        ceil_x[equal_mask] = torch.clamp(ceil_x[equal_mask] + 1, max=26)
        # Update the indicator_map with the adjusted values
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        fx = fx * indicator_map.float()

        return fx.unsqueeze(1), indicator_map.unsqueeze(1), indices, weights_lower, weights_upper

    def monotonicity_loss(self, output):
        # Assuming output is of shape [batch_size, ..., sequence_length]
        diffs = output[..., 1:] - output[..., :-1]
        penalty = torch.mean(F.relu(-diffs))  # Penalize negative differences
        return penalty

    def forward_loss(self, label, output, mask_label, fx, indicator_map):
        s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
        s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))

        meta_loss1 = torch.abs(output - fx) * indicator_map * mask_label
        meta_loss2 = (output - fx) ** 2 * indicator_map * mask_label

        if self.rescale_loss:
            meta_loss_mask = (meta_loss1 > self.range).float()
            meta_loss1 = meta_loss1 * meta_loss_mask
            meta_loss2 = meta_loss2 * meta_loss_mask

        meta_loss = torch.mean(meta_loss1 * self.lambda_g1v + meta_loss2 * self.lambda_g2v)

        # tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        # tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        monotonicity_loss = self.monotonicity_loss(output)

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + monotonicity_loss * self.lambda_g3v + meta_loss * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        fx, indicator_map, indices, weights_lower, weights_upper = self.calculate_fx_and_indicator_map(meta_data.detach(), meta_data_mask.detach())

        if self.meta_feature:
            common_part = meta_data[:, :, :, :2].expand(-1, 1, -1, -1)
            time_fx = torch.cat([common_part, fx], dim=-1)
            common_part = meta_data_mask[:, :, :, :2].expand(-1, 1, -1, -1)
            time_indicator_map = torch.cat([common_part, indicator_map], dim=-1)
            meta, meta_mask = self.meta_net(time_fx, time_indicator_map)
            h = h * m + meta * meta_mask

            if self.meta_dim > 1:
                common_part = meta_data[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data[:, :, :, 2:-1].transpose(1, 3)
                omni = torch.cat([common_part, channels], dim=-1)
                common_part = meta_data_mask[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data_mask[:, :, :, 2:-1].transpose(1, 3)
                omni_mask = torch.cat([common_part, channels], dim=-1)
                omni, omni_mask = self.meta_net2(omni, omni_mask)

                h = h + omni * omni_mask

        h = self.encoder(h)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label, fx, indicator_map)

        return loss, output


class Fluex_net_meta_geo(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., meta_dim=1, rescale_loss=False, step_two=False, res=True, means=4.05,
                 stds=1.14, meta_feature=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.shell_num = shell_num
        self.res = res
        self.drop_sat = drop_sat
        self.meta_dim = meta_dim
        self.meta_feature = meta_feature
        self.means = means
        self.stds = stds

        self.p = (np.log10(1500) - means) / stds
        self.q = (np.log10(4500) - means) / stds
        self.m = (np.log10(3000) - means) / stds
        self.p, self.q, self.m = self.p.item(), self.q.item(), self.m.item()
        self.range = self.q - self.p

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans-1, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans - 1
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.omni_chans = 8

        self.meta_net, self.meta_net2 = None, None
        if meta_feature:
            self.meta_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)
            if meta_dim > 1:
                self.meta_net2 = Encoder_sat(meta_dim-1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)

        self.geo_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim, out_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def float_to_index_and_weights(self, float_mask):
        indices = ((float_mask - 2.8) / 0.2)
        fractions = ((float_mask - 2.8) / 0.2) % 1

        weights_lower = 1 - fractions
        weights_upper = fractions

        # Handle edge cases (less than 2.8 and greater than 8.0)
        less_mask = float_mask < 2.8
        more_mask = float_mask > 8.0

        indices[less_mask] = 0
        indices[more_mask] = 26  # this is because 8.0 corresponds to the 27th index (0-indexed)

        weights_lower[less_mask | more_mask] = 1.0
        weights_upper[less_mask | more_mask] = 0.0

        return indices, weights_lower, weights_upper

    def calculate_fx_and_indicator_map(self, meta_data, meta_data_mask):
        meta_data = meta_data[:, 0, :, -1]
        meta_data_mask = meta_data_mask[:, 0, :, -1]
        float_mask = meta_data * (1 - meta_data_mask)
        indices, weights_lower, weights_upper = self.float_to_index_and_weights(float_mask)

        x0_tensor = indices.unsqueeze(-1)
        B, T, _ = x0_tensor.shape

        # Generate x values from 0 to 25 (27 elements)
        x_values = torch.arange(0, 27, dtype=x0_tensor.dtype, device=x0_tensor.device)

        # Calculate f(x) for each x in x_values
        # fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)
        fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)

        # Calculate floor and ceil for each x0
        floor_x = torch.floor(x0_tensor).to(torch.int64)
        ceil_x = torch.ceil(x0_tensor).to(torch.int64)

        # Create indicator map (initialize with False)
        indicator_map = torch.zeros(B, T, 27, dtype=torch.bool, device=x0_tensor.device)

        # Set True at positions corresponding to floor(x) and ceil(x)
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        equal_mask = floor_x == ceil_x
        floor_x[equal_mask] = torch.clamp(floor_x[equal_mask] - 1, min=0)
        ceil_x[equal_mask] = torch.clamp(ceil_x[equal_mask] + 1, max=26)
        # Update the indicator_map with the adjusted values
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        fx = fx * indicator_map.float()

        return fx.unsqueeze(1), indicator_map.unsqueeze(1), indices, weights_lower, weights_upper

    def monotonicity_loss(self, output):
        # Assuming output is of shape [batch_size, ..., sequence_length]
        diffs = output[..., 1:] - output[..., :-1]
        penalty = torch.mean(F.relu(-diffs))  # Penalize negative differences
        return penalty

    def forward_loss(self, label, output, mask_label, fx, indicator_map):
        s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
        s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))

        meta_loss1 = torch.abs(output - fx) * indicator_map * mask_label
        meta_loss2 = (output - fx) ** 2 * indicator_map * mask_label

        if self.rescale_loss:
            meta_loss_mask = (meta_loss1 > self.range).float()
            meta_loss1 = meta_loss1 * meta_loss_mask
            meta_loss2 = meta_loss2 * meta_loss_mask

        meta_loss = torch.mean(meta_loss1 * self.lambda_g1v + meta_loss2 * self.lambda_g2v)

        # tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        # tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        monotonicity_loss = self.monotonicity_loss(output)

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + monotonicity_loss * self.lambda_g3v + meta_loss * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        if self.drop_sat >= 0:
            len_keep = int(self.num_sat * (1 - self.drop_sat))
        elif (not self.training) and self.drop_sat < 0:
            len_keep = int(self.num_sat)
        else:
            len_keep = random.randint(3, 12)
        noise = torch.rand(self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_keep = ids_shuffle[:len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep:  # or (not self.training)
                output, update_mask = encoder(inputs[:, i, :-1], mask[:, i, :-1])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        geo, geo_mask = self.geo_net(inputs[:, 0, -1:], mask[:, 0, -1:])

        fx, indicator_map, indices, weights_lower, weights_upper = self.calculate_fx_and_indicator_map(meta_data.detach(), meta_data_mask.detach())

        if self.meta_feature:
            common_part = meta_data[:, :, :, :2].expand(-1, 1, -1, -1)
            time_fx = torch.cat([common_part, fx], dim=-1)
            common_part = meta_data_mask[:, :, :, :2].expand(-1, 1, -1, -1)
            time_indicator_map = torch.cat([common_part, indicator_map], dim=-1)
            meta, meta_mask = self.meta_net(time_fx, time_indicator_map)
            h = h * m + meta * meta_mask + geo * geo_mask

            if self.meta_dim > 1:
                common_part = meta_data[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data[:, :, :, 2:-1].transpose(1, 3)
                omni = torch.cat([common_part, channels], dim=-1)
                common_part = meta_data_mask[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data_mask[:, :, :, 2:-1].transpose(1, 3)
                omni_mask = torch.cat([common_part, channels], dim=-1)
                omni, omni_mask = self.meta_net2(omni, omni_mask)

                h = h + omni * omni_mask

        h = self.encoder(h)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label, fx, indicator_map)

        return loss, output


class Fluex_net_meta_geolater(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., meta_dim=1, rescale_loss=False, step_two=False, res=True, means=4.05,
                 stds=1.14, meta_feature=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.shell_num = shell_num
        self.res = res
        self.drop_sat = drop_sat
        self.meta_dim = meta_dim
        self.meta_feature = meta_feature
        self.means = means
        self.stds = stds

        self.p = (np.log10(1500) - means) / stds
        self.q = (np.log10(4500) - means) / stds
        self.m = (np.log10(3000) - means) / stds
        self.p, self.q, self.m = self.p.item(), self.q.item(), self.m.item()
        self.range = self.q - self.p

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans-1, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans - 1
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.omni_chans = 8

        self.meta_net, self.meta_net2 = None, None
        if meta_feature:
            self.meta_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)
            if meta_dim > 1:
                self.meta_net2 = Encoder_sat(meta_dim-1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim, out_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        latent_dim = 16
        self.geo_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2,), dropout=drop_path, temb_channels=shell_num)
        self.output_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2,), dropout=drop_path, temb_channels=shell_num)
        self.conv_funsion = nn.Sequential(
                                            torch.nn.GroupNorm(num_groups=4, num_channels=latent_dim, eps=1e-6, affine=True),
                                            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1),
                                            torch.nn.GroupNorm(num_groups=4, num_channels=latent_dim, eps=1e-6, affine=True),
                                            nn.GELU(),
                                            nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)
                                        )
        # self.geo_encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim, out_chans=latent_dim,
        #                                embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        #                                mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)
        # self.norm_out2 = torch.nn.GroupNorm(num_groups=4, num_channels=latent_dim, eps=1e-6, affine=True)
        # self.conv_out2 = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def float_to_index_and_weights(self, float_mask):
        indices = ((float_mask - 2.8) / 0.2)
        fractions = ((float_mask - 2.8) / 0.2) % 1

        weights_lower = 1 - fractions
        weights_upper = fractions

        # Handle edge cases (less than 2.8 and greater than 8.0)
        less_mask = float_mask < 2.8
        more_mask = float_mask > 8.0

        indices[less_mask] = 0
        indices[more_mask] = 26  # this is because 8.0 corresponds to the 27th index (0-indexed)

        weights_lower[less_mask | more_mask] = 1.0
        weights_upper[less_mask | more_mask] = 0.0

        return indices, weights_lower, weights_upper

    def calculate_fx_and_indicator_map(self, meta_data, meta_data_mask):
        meta_data = meta_data[:, 0, :, -1]
        meta_data_mask = meta_data_mask[:, 0, :, -1]
        float_mask = meta_data * (1 - meta_data_mask)
        indices, weights_lower, weights_upper = self.float_to_index_and_weights(float_mask)

        x0_tensor = indices.unsqueeze(-1)
        B, T, _ = x0_tensor.shape

        # Generate x values from 0 to 25 (27 elements)
        x_values = torch.arange(0, 27, dtype=x0_tensor.dtype, device=x0_tensor.device)

        # Calculate f(x) for each x in x_values
        # fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)
        fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)

        # Calculate floor and ceil for each x0
        floor_x = torch.floor(x0_tensor).to(torch.int64)
        ceil_x = torch.ceil(x0_tensor).to(torch.int64)

        # Create indicator map (initialize with False)
        indicator_map = torch.zeros(B, T, 27, dtype=torch.bool, device=x0_tensor.device)

        # Set True at positions corresponding to floor(x) and ceil(x)
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        equal_mask = floor_x == ceil_x
        floor_x[equal_mask] = torch.clamp(floor_x[equal_mask] - 1, min=0)
        ceil_x[equal_mask] = torch.clamp(ceil_x[equal_mask] + 1, max=26)
        # Update the indicator_map with the adjusted values
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        fx = fx * indicator_map.float()

        return fx.unsqueeze(1), indicator_map.unsqueeze(1), indices, weights_lower, weights_upper

    def monotonicity_loss(self, output):
        # Assuming output is of shape [batch_size, ..., sequence_length]
        diffs = output[..., 1:] - output[..., :-1]
        penalty = torch.mean(F.relu(-diffs))  # Penalize negative differences
        return penalty

    def forward_loss(self, label, output, mask_label, fx, indicator_map):
        s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
        s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))

        meta_loss1 = torch.abs(output - fx) * indicator_map * mask_label
        meta_loss2 = (output - fx) ** 2 * indicator_map * mask_label

        if self.rescale_loss:
            meta_loss_mask = (meta_loss1 > self.range).float()
            meta_loss1 = meta_loss1 * meta_loss_mask
            meta_loss2 = meta_loss2 * meta_loss_mask

        meta_loss = torch.mean(meta_loss1 * self.lambda_g1v + meta_loss2 * self.lambda_g2v)

        # tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        # tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        monotonicity_loss = self.monotonicity_loss(output)

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + monotonicity_loss * self.lambda_g3v + meta_loss * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        if self.drop_sat >= 0:
            len_keep = int(self.num_sat * (1 - self.drop_sat))
        elif (not self.training) and self.drop_sat < 0:
            len_keep = int(self.num_sat)
        else:
            len_keep = random.randint(3, 12)
        noise = torch.rand(self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_keep = ids_shuffle[:len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep:  # or (not self.training)
                output, update_mask = encoder(inputs[:, i, :-1], mask[:, i, :-1])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        fx, indicator_map, indices, weights_lower, weights_upper = self.calculate_fx_and_indicator_map(meta_data.detach(), meta_data_mask.detach())

        if self.meta_feature:
            common_part = meta_data[:, :, :, :2].expand(-1, 1, -1, -1)
            time_fx = torch.cat([common_part, fx], dim=-1)
            common_part = meta_data_mask[:, :, :, :2].expand(-1, 1, -1, -1)
            time_indicator_map = torch.cat([common_part, indicator_map], dim=-1)
            meta, meta_mask = self.meta_net(time_fx, time_indicator_map)
            h = h * m + meta * meta_mask

            if self.meta_dim > 1:
                common_part = meta_data[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data[:, :, :, 2:-1].transpose(1, 3)
                omni = torch.cat([common_part, channels], dim=-1)
                common_part = meta_data_mask[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data_mask[:, :, :, 2:-1].transpose(1, 3)
                omni_mask = torch.cat([common_part, channels], dim=-1)
                omni, omni_mask = self.meta_net2(omni, omni_mask)

                h = h + omni * omni_mask

        h = self.encoder(h)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label, fx, indicator_map)

        geo, geo_mask = self.geo_net(inputs[:, 0, -1:, :, 16:], mask[:, 0, -1:, :, 16:])
        o = torch.cat([inputs[:, 0, 0:1, :, :2], output[:, :, :, :16].detach()], dim=-1)
        h, m = self.output_net(o.detach(), torch.ones_like(o, device=output.device))
        final_output = torch.cat([h, geo * geo_mask], dim=-1)
        final_output = self.conv_funsion(final_output)
        # final_output = self.geo_encoder(final_output)
        # final_output = self.norm_out2(final_output)
        # final_output = F.gelu(final_output)
        # final_output = self.conv_out2(final_output)

        loss2 = self.forward_loss(label, final_output, mask_label, fx, indicator_map)

        return loss+loss2, final_output


class Fluex_net_meta_geolater_old(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., meta_dim=1, rescale_loss=False, step_two=False, res=True, means=4.05,
                 stds=1.14, meta_feature=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.shell_num = shell_num
        self.res = res
        self.drop_sat = drop_sat
        self.meta_dim = meta_dim
        self.meta_feature = meta_feature
        self.means = means
        self.stds = stds

        self.p = (np.log10(1500) - means) / stds
        self.q = (np.log10(4500) - means) / stds
        self.m = (np.log10(3000) - means) / stds
        self.p, self.q, self.m = self.p.item(), self.q.item(), self.m.item()
        self.range = self.q - self.p

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans-1, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans - 1
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.omni_chans = 8

        self.meta_net, self.meta_net2 = None, None
        if meta_feature:
            self.meta_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)
            if meta_dim > 1:
                self.meta_net2 = Encoder_sat(meta_dim-1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim, out_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        self.geo_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)
        self.output_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)
        self.geo_encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim, out_chans=latent_dim,
                                       embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                       mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)
        self.norm_out2 = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out2 = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def float_to_index_and_weights(self, float_mask):
        indices = ((float_mask - 2.8) / 0.2)
        fractions = ((float_mask - 2.8) / 0.2) % 1

        weights_lower = 1 - fractions
        weights_upper = fractions

        # Handle edge cases (less than 2.8 and greater than 8.0)
        less_mask = float_mask < 2.8
        more_mask = float_mask > 8.0

        indices[less_mask] = 0
        indices[more_mask] = 26  # this is because 8.0 corresponds to the 27th index (0-indexed)

        weights_lower[less_mask | more_mask] = 1.0
        weights_upper[less_mask | more_mask] = 0.0

        return indices, weights_lower, weights_upper

    def calculate_fx_and_indicator_map(self, meta_data, meta_data_mask):
        meta_data = meta_data[:, 0, :, -1]
        meta_data_mask = meta_data_mask[:, 0, :, -1]
        float_mask = meta_data * (1 - meta_data_mask)
        indices, weights_lower, weights_upper = self.float_to_index_and_weights(float_mask)

        x0_tensor = indices.unsqueeze(-1)
        B, T, _ = x0_tensor.shape

        # Generate x values from 0 to 25 (27 elements)
        x_values = torch.arange(0, 27, dtype=x0_tensor.dtype, device=x0_tensor.device)

        # Calculate f(x) for each x in x_values
        # fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)
        fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)

        # Calculate floor and ceil for each x0
        floor_x = torch.floor(x0_tensor).to(torch.int64)
        ceil_x = torch.ceil(x0_tensor).to(torch.int64)

        # Create indicator map (initialize with False)
        indicator_map = torch.zeros(B, T, 27, dtype=torch.bool, device=x0_tensor.device)

        # Set True at positions corresponding to floor(x) and ceil(x)
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        equal_mask = floor_x == ceil_x
        floor_x[equal_mask] = torch.clamp(floor_x[equal_mask] - 1, min=0)
        ceil_x[equal_mask] = torch.clamp(ceil_x[equal_mask] + 1, max=26)
        # Update the indicator_map with the adjusted values
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        fx = fx * indicator_map.float()

        return fx.unsqueeze(1), indicator_map.unsqueeze(1), indices, weights_lower, weights_upper

    def monotonicity_loss(self, output):
        # Assuming output is of shape [batch_size, ..., sequence_length]
        diffs = output[..., 1:] - output[..., :-1]
        penalty = torch.mean(F.relu(-diffs))  # Penalize negative differences
        return penalty

    def forward_loss(self, label, output, mask_label, fx, indicator_map):
        s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
        s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))

        meta_loss1 = torch.abs(output - fx) * indicator_map * mask_label
        meta_loss2 = (output - fx) ** 2 * indicator_map * mask_label

        if self.rescale_loss:
            meta_loss_mask = (meta_loss1 > self.range).float()
            meta_loss1 = meta_loss1 * meta_loss_mask
            meta_loss2 = meta_loss2 * meta_loss_mask

        meta_loss = torch.mean(meta_loss1 * self.lambda_g1v + meta_loss2 * self.lambda_g2v)

        # tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        # tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        monotonicity_loss = self.monotonicity_loss(output)

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + monotonicity_loss * self.lambda_g3v + meta_loss * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        if self.drop_sat >= 0:
            len_keep = int(self.num_sat * (1 - self.drop_sat))
        elif (not self.training) and self.drop_sat < 0:
            len_keep = int(self.num_sat)
        else:
            len_keep = random.randint(3, 12)
        noise = torch.rand(self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_keep = ids_shuffle[:len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep:  # or (not self.training)
                output, update_mask = encoder(inputs[:, i, :-1], mask[:, i, :-1])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        fx, indicator_map, indices, weights_lower, weights_upper = self.calculate_fx_and_indicator_map(meta_data.detach(), meta_data_mask.detach())

        if self.meta_feature:
            common_part = meta_data[:, :, :, :2].expand(-1, 1, -1, -1)
            time_fx = torch.cat([common_part, fx], dim=-1)
            common_part = meta_data_mask[:, :, :, :2].expand(-1, 1, -1, -1)
            time_indicator_map = torch.cat([common_part, indicator_map], dim=-1)
            meta, meta_mask = self.meta_net(time_fx, time_indicator_map)
            h = h * m + meta * meta_mask

            if self.meta_dim > 1:
                common_part = meta_data[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data[:, :, :, 2:-1].transpose(1, 3)
                omni = torch.cat([common_part, channels], dim=-1)
                common_part = meta_data_mask[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data_mask[:, :, :, 2:-1].transpose(1, 3)
                omni_mask = torch.cat([common_part, channels], dim=-1)
                omni, omni_mask = self.meta_net2(omni, omni_mask)

                h = h + omni * omni_mask

        h = self.encoder(h)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label, fx, indicator_map)

        geo, geo_mask = self.geo_net(inputs[:, 0, -1:], mask[:, 0, -1:])
        o = torch.cat([inputs[:, 0, 0:1, :, :2], output.detach()], dim=-1)
        h, m = self.output_net(o.detach(), torch.ones_like(o, device=output.device))
        final_output = self.geo_encoder(geo * geo_mask + h * m)
        final_output = self.norm_out2(final_output)
        final_output = F.gelu(final_output)
        final_output = self.conv_out2(final_output)

        loss2 = self.forward_loss(label, final_output, mask_label, fx, indicator_map)

        return loss+loss2, final_output


class Fluex_net_meta_TwoStep(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., meta_dim=1, rescale_loss=False, step_two=False, res=True, means=4.05,
                 stds=1.14, meta_feature=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.shell_num = shell_num
        self.res = res
        self.drop_sat = drop_sat
        self.meta_dim = meta_dim
        self.meta_feature = meta_feature
        self.means = means
        self.stds = stds

        self.p = (np.log10(1500) - means) / stds
        self.q = (np.log10(4500) - means) / stds
        self.m = (np.log10(3000) - means) / stds
        self.p, self.q, self.m = self.p.item(), self.q.item(), self.m.item()
        self.range = self.q - self.p

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans - 1
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.omni_chans = 8

        self.meta_net, self.meta_net2 = None, None
        if meta_feature:
            self.meta_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)
            if meta_dim > 1:
                self.meta_net2 = Encoder_sat(meta_dim-1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)

        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=latent_dim, out_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        self.conv_meta = nn.Sequential(
                            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
                            act_layer(),
                            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
                            act_layer(),
                            nn.Conv2d(2, 2, kernel_size=(3,1), stride=1, padding=(1,0)),
                            act_layer(),
                            nn.Conv2d(2, 1, kernel_size=(3,1), stride=1, padding=(1,0)),
                        )

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def float_to_index_and_weights(self, float_mask):
        indices = ((float_mask - 2.8) / 0.2)
        fractions = ((float_mask - 2.8) / 0.2) % 1

        weights_lower = 1 - fractions
        weights_upper = fractions

        # Handle edge cases (less than 2.8 and greater than 8.0)
        less_mask = float_mask < 2.8
        more_mask = float_mask > 8.0

        indices[less_mask] = 0
        indices[more_mask] = 26  # this is because 8.0 corresponds to the 27th index (0-indexed)

        weights_lower[less_mask | more_mask] = 1.0
        weights_upper[less_mask | more_mask] = 0.0

        return indices, weights_lower, weights_upper

    def calculate_fx_and_indicator_map(self, meta_data, meta_data_mask):
        meta_data = meta_data[:, 0, :, -1]
        meta_data_mask = meta_data_mask[:, 0, :, -1]
        float_mask = meta_data * (1 - meta_data_mask)
        indices, weights_lower, weights_upper = self.float_to_index_and_weights(float_mask)

        x0_tensor = indices.unsqueeze(-1)
        B, T, _ = x0_tensor.shape

        # Generate x values from 0 to 25 (27 elements)
        x_values = torch.arange(0, 27, dtype=x0_tensor.dtype, device=x0_tensor.device)

        # Calculate f(x) for each x in x_values
        # fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)
        fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)

        # Calculate floor and ceil for each x0
        floor_x = torch.floor(x0_tensor).to(torch.int64)
        ceil_x = torch.ceil(x0_tensor).to(torch.int64)

        # Create indicator map (initialize with False)
        indicator_map = torch.zeros(B, T, 27, dtype=torch.bool, device=x0_tensor.device)

        # Set True at positions corresponding to floor(x) and ceil(x)
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        equal_mask = floor_x == ceil_x
        floor_x[equal_mask] = torch.clamp(floor_x[equal_mask] - 1, min=0)
        ceil_x[equal_mask] = torch.clamp(ceil_x[equal_mask] + 1, max=26)
        # Update the indicator_map with the adjusted values
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        fx = fx * indicator_map.float()

        return fx.unsqueeze(1), indicator_map.unsqueeze(1), indices, weights_lower, weights_upper

    def monotonicity_loss(self, output):
        # Assuming output is of shape [batch_size, ..., sequence_length]
        diffs = output[..., 1:] - output[..., :-1]
        penalty = torch.mean(F.relu(-diffs))  # Penalize negative differences
        return penalty

    def forward_loss(self, label, output, mask_label, fx, indicator_map):
        s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
        s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))

        meta_loss1 = torch.abs(output - fx) * indicator_map * mask_label
        meta_loss2 = (output - fx) ** 2 * indicator_map * mask_label

        if self.rescale_loss:
            meta_loss_mask = (meta_loss1 > self.range).float()
            meta_loss1 = meta_loss1 * meta_loss_mask
            meta_loss2 = meta_loss2 * meta_loss_mask

        meta_loss = torch.mean(meta_loss1 * self.lambda_g1v + meta_loss2 * self.lambda_g2v)

        # tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        # tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        monotonicity_loss = self.monotonicity_loss(output)

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + monotonicity_loss * self.lambda_g3v + meta_loss * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        fx, indicator_map, indices, weights_lower, weights_upper = self.calculate_fx_and_indicator_map(meta_data.detach(), meta_data_mask.detach())

        if self.meta_feature:
            common_part = meta_data[:, :, :, :2].expand(-1, 1, -1, -1)
            time_fx = torch.cat([common_part, fx], dim=-1)
            common_part = meta_data_mask[:, :, :, :2].expand(-1, 1, -1, -1)
            time_indicator_map = torch.cat([common_part, indicator_map], dim=-1)
            meta, meta_mask = self.meta_net(time_fx, time_indicator_map)
            h = h * m + meta * meta_mask

            if self.meta_dim > 1:
                common_part = meta_data[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data[:, :, :, 2:-1].transpose(1, 3)
                omni = torch.cat([common_part, channels], dim=-1)
                common_part = meta_data_mask[:, :, :, :2].expand(-1, self.meta_dim-1, -1, -1)
                channels = meta_data_mask[:, :, :, 2:-1].transpose(1, 3)
                omni_mask = torch.cat([common_part, channels], dim=-1)
                omni, omni_mask = self.meta_net2(omni, omni_mask)

                h = h + omni * omni_mask

        h = self.encoder(h)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        output = self.conv_meta(torch.cat([output, fx], dim=1))

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label, fx, indicator_map)

        return loss, output


class Fluex_net_meta_omni(nn.Module):
    """ same but more general than 1
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, latent_dim=256, num_heads=16, drop_path=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU, lambda_g1v=0., lambda_g2v=1., lambda_g3v=0., lambda_g4v=0.,
                 num_sat=2, shell_num=2, num_res_blocks=1, drop_sat=0., meta_dim=1, rescale_loss=False, step_two=False, res=True, means=4.05,
                 stds=1.14, meta_feature=True):
        super().__init__()
        self.lambda_g1v = lambda_g1v
        self.lambda_g2v = lambda_g2v
        self.lambda_g3v = lambda_g3v
        self.lambda_g4v = lambda_g4v
        self.num_sat = num_sat
        self.step_two = step_two
        self.shell_num = shell_num
        self.res = res
        self.drop_sat = drop_sat
        self.meta_dim = meta_dim
        self.meta_feature = meta_feature
        self.means = means
        self.stds = stds

        self.p = (np.log10(1500) - means) / stds
        self.q = (np.log10(4500) - means) / stds
        self.m = (np.log10(3000) - means) / stds
        self.p, self.q, self.m = self.p.item(), self.q.item(), self.m.item()
        self.range = self.q - self.p

        self.rescale_loss = rescale_loss

        self.sat_blocks = nn.ModuleList()
        for i in range(num_sat):
            self.sat_blocks.append(Encoder_sat(in_chans, latent_dim, 32, num_res_blocks, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num))

        # --------------------------------------------------------------------------
        # Seismic MAE encoder specifics
        img_size = to_2tuple(img_size)
        img_size = (img_size[0], img_size[1] - shell_num)
        patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans - 1
        self.embed_dim = embed_dim
        self.num_patch = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.omni_chans = 8

        self.meta_net, self.meta_net2 = None, None
        if meta_feature:
            self.meta_net = Encoder_sat(1, latent_dim, 32, num_res_blocks + 2, ch_mult=(2, 4), dropout=drop_path, temb_channels=shell_num)
            if meta_dim > 1:
                self.meta_net2 = Encoder_all(img_size=(img_size[0], 1), patch_size=(9, 1), in_chans=meta_dim-1, out_chans=self.omni_chans,
                                    embed_dim=self.omni_chans, depth=2, num_heads=2,
                                    mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        all_in_chans = latent_dim if meta_dim <= 1 else latent_dim + self.omni_chans
        self.encoder = Encoder_all(img_size=img_size, patch_size=patch_size, in_chans=all_in_chans, out_chans=latent_dim,
                                   embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                   mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, drop_path=drop_path)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=latent_dim, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(latent_dim, 1, kernel_size=3, stride=1, padding=1)

        if self.step_two:
            self.out_shell_net = nn.Linear(9, 4)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def float_to_index_and_weights(self, float_mask):
        indices = ((float_mask - 2.8) / 0.2)
        fractions = ((float_mask - 2.8) / 0.2) % 1

        weights_lower = 1 - fractions
        weights_upper = fractions

        # Handle edge cases (less than 2.8 and greater than 8.0)
        less_mask = float_mask < 2.8
        more_mask = float_mask > 8.0

        indices[less_mask] = 0
        indices[more_mask] = 26  # this is because 8.0 corresponds to the 27th index (0-indexed)

        weights_lower[less_mask | more_mask] = 1.0
        weights_upper[less_mask | more_mask] = 0.0

        return indices, weights_lower, weights_upper

    def calculate_fx_and_indicator_map(self, meta_data, meta_data_mask):
        meta_data = meta_data[:, 0, :, -1]
        meta_data_mask = meta_data_mask[:, 0, :, -1]
        float_mask = meta_data * (1 - meta_data_mask)
        indices, weights_lower, weights_upper = self.float_to_index_and_weights(float_mask)

        x0_tensor = indices.unsqueeze(-1)
        B, T, _ = x0_tensor.shape

        # Generate x values from 0 to 25 (27 elements)
        x_values = torch.arange(0, 27, dtype=x0_tensor.dtype, device=x0_tensor.device)

        # Calculate f(x) for each x in x_values
        # fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)
        fx = self.p * x_values.unsqueeze(0).unsqueeze(0) + (self.m - self.p * x0_tensor)

        # Calculate floor and ceil for each x0
        floor_x = torch.floor(x0_tensor).to(torch.int64)
        ceil_x = torch.ceil(x0_tensor).to(torch.int64)

        # Create indicator map (initialize with False)
        indicator_map = torch.zeros(B, T, 27, dtype=torch.bool, device=x0_tensor.device)

        # Set True at positions corresponding to floor(x) and ceil(x)
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        equal_mask = floor_x == ceil_x
        floor_x[equal_mask] = torch.clamp(floor_x[equal_mask] - 1, min=0)
        ceil_x[equal_mask] = torch.clamp(ceil_x[equal_mask] + 1, max=26)
        # Update the indicator_map with the adjusted values
        indicator_map.scatter_(2, floor_x, True)  # Set True at floor(x) positions
        indicator_map.scatter_(2, ceil_x, True)  # Set True at ceil(x) positions

        fx = fx * indicator_map.float()

        return fx.unsqueeze(1), indicator_map.unsqueeze(1), indices, weights_lower, weights_upper

    def monotonicity_loss(self, output):
        # Assuming output is of shape [batch_size, ..., sequence_length]
        diffs = output[..., 1:] - output[..., :-1]
        penalty = torch.mean(F.relu(-diffs))  # Penalize negative differences
        return penalty

    def forward_loss(self, label, output, mask_label, fx, indicator_map):
        s2v_loss1 = torch.mean(torch.abs(output - label) * (1 - mask_label))
        s2v_loss2 = torch.mean((output - label) ** 2 * (1 - mask_label))

        meta_loss1 = torch.abs(output - fx) * indicator_map * mask_label
        meta_loss2 = (output - fx) ** 2 * indicator_map * mask_label

        if self.rescale_loss:
            meta_loss_mask = (meta_loss1 > self.range).float()
            meta_loss1 = meta_loss1 * meta_loss_mask
            meta_loss2 = meta_loss2 * meta_loss_mask

        meta_loss = torch.mean(meta_loss1 * self.lambda_g1v + meta_loss2 * self.lambda_g2v)

        # tv_h = torch.pow((output[:, :, 1:, :] - output[:, :, :-1, :]), 2).sum()
        # tv_w = torch.pow((output[:, :, :, 1:] - output[:, :, :, :-1]), 2).sum()

        monotonicity_loss = self.monotonicity_loss(output)

        loss = s2v_loss1 * self.lambda_g1v + s2v_loss2 * self.lambda_g2v \
               + monotonicity_loss * self.lambda_g3v + meta_loss * self.lambda_g4v

        return loss

    def forward(self, inputs, label, mask, mask_label, meta_data, meta_data_mask):
        mask = 1 - mask  # 1 for validated value, 0 for nan
        h, m = [], []

        len_keep = int(self.num_sat * (1 - self.drop_sat))
        noise = torch.rand(inputs.shape[0], self.num_sat)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        for i, encoder in enumerate(self.sat_blocks):
            if i in ids_keep or (not self.training):
                output, update_mask = encoder(inputs[:, i], mask[:, i])
                h.append(output)
                m.append(update_mask)
        h = torch.mean(torch.stack(h), dim=0)
        m = torch.clamp(torch.sum(torch.stack(m), dim=0), 0, 1)

        fx, indicator_map, indices, weights_lower, weights_upper = self.calculate_fx_and_indicator_map(meta_data.detach(), meta_data_mask.detach())

        if self.meta_feature:
            common_part = meta_data[:, :, :, :2].expand(-1, 1, -1, -1)
            time_fx = torch.cat([common_part, fx], dim=-1)
            common_part = meta_data_mask[:, :, :, :2].expand(-1, 1, -1, -1)
            time_indicator_map = torch.cat([common_part, indicator_map], dim=-1)
            meta, meta_mask = self.meta_net(time_fx, time_indicator_map)
            h = h * m + meta * meta_mask

            if self.meta_dim > 1:
                omni = meta_data[:, :, :, 2:-1].transpose(1, 3)
                omni_mask = meta_data_mask[:, :, :, 2:-1].transpose(1, 3)
                omni = self.meta_net2(omni*omni_mask)
                h = torch.cat([h, omni.repeat((1, 1, 1, h.shape[-1]))], dim=1)

        h = self.encoder(h)

        h = self.norm_out(h)
        h = F.gelu(h)
        output = self.conv_out(h)

        if self.step_two:
            B, C, T, W = output.shape
            output2 = self.out_shell_net(output[:, :, :, 5:14].detach().reshape(B, C * T, 9)).reshape(B, C, T, 4)
            output = torch.cat([output[:, :, :, 0:1], output2[:, :, :, :], output[:, :, :, 5:]], dim=-1)

        loss = self.forward_loss(label, output, mask_label, fx, indicator_map)

        return loss, output


################# Functions ########################


def FluexNet(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
             mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0., lambda_g4v=0.,
             num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, **kwargs):
    model = Fluex_net(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                      num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                      mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                      lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                      res=res)
    return model


def FluexNet_cnnonly(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
             mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0., lambda_g4v=0.,
             num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, **kwargs):
    model = Fluex_net_cnnonly(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                      num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                      mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                      lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                      res=res)
    return model



def FluexNet_small(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                   mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                   lambda_g4v=0., drop_sat=0.,
                   num_sat=12, shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, **kwargs):
    model = Fluex_net_small(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                            num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                            mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                            lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                            res=res)
    return model


def FluexNet_single(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                    mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                    lambda_g4v=0., drop_sat=0.,
                    num_sat=12, shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, target_shell=0, **kwargs):
    model = Fluex_net_single(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                             num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                             mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                             lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                             res=res,
                             target_shell=target_shell)
    return model


def FluexNet_mlp(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                 lambda_g4v=0., drop_sat=0.,
                 num_sat=12, shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, target_shell=0, **kwargs):
    model = Fluex_net_mlp(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                          num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                          mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                          lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                          res=res)
    return model


def FluexNet_omni(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                  lambda_g4v=0.,
                  num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, **kwargs):
    model = Fluex_net_omni(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                           num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                           lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                           res=res)
    return model


def FluexNet_avg(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                 lambda_g4v=0.,
                 num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, **kwargs):
    model = Fluex_net_avg(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                          num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                          mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                          lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                          res=res)
    return model


def FluexNet_argument(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                      mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                      lambda_g4v=0.,
                      num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, **kwargs):
    model = Fluex_net_argument(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                               num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                               mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                               lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num,
                               step_two=step_two,
                               res=res)
    return model


def FluexNet_label(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                   mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                   lambda_g4v=0.,
                   num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, **kwargs):
    model = Fluex_net_label(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                            num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                            mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                            lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                            res=res)
    return model


def FluexNet_gradient(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                      mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                      lambda_g4v=0.,
                      num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, **kwargs):
    model = Fluex_net_gradient(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                               num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                               mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                               lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num,
                               step_two=step_two,
                               res=res)
    return model


def FluexNet_meta_old(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                      mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                      lambda_g4v=0.,
                      num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, meta_dim=1, means=4.05,
                      stds=1.14, **kwargs):
    model = Fluex_net_meta_old(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                               num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                               mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                               lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num,
                               step_two=step_two,
                               res=res, meta_dim=meta_dim, means=means, stds=stds)
    return model


def FluexNet_meta(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                  lambda_g4v=0.,
                  num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, meta_dim=1, means=4.05,
                  stds=1.14, meta_feature=True, **kwargs):
    model = Fluex_net_meta(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                           num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                           lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                           res=res, meta_dim=meta_dim, means=means, stds=stds, meta_feature=meta_feature)
    return model


def FluexNet_meta_future(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                  lambda_g4v=0.,
                  num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, meta_dim=1, means=4.05,
                  stds=1.14, meta_feature=True, **kwargs):
    model = Fluex_net_meta_future(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                           num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                           lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                           res=res, meta_dim=meta_dim, means=means, stds=stds, meta_feature=meta_feature)
    return model


def FluexNet_meta2(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                  lambda_g4v=0.,
                  num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, meta_dim=1, means=4.05,
                  stds=1.14, meta_feature=True, **kwargs):
    model = Fluex_net_meta2(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                           num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                           lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                           res=res, meta_dim=meta_dim, means=means, stds=stds, meta_feature=meta_feature)
    return model

def FluexNet_meta_geo(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                  lambda_g4v=0.,
                  num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, meta_dim=1, means=4.05,
                  stds=1.14, meta_feature=True, **kwargs):
    model = Fluex_net_meta_geo(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                           num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                           lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                           res=res, meta_dim=meta_dim, means=means, stds=stds, meta_feature=meta_feature)
    return model


def FluexNet_meta_geolater(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                  lambda_g4v=0.,
                  num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, meta_dim=1, means=4.05,
                  stds=1.14, meta_feature=True, **kwargs):
    model = Fluex_net_meta_geolater(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                           num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                           lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                           res=res, meta_dim=meta_dim, means=means, stds=stds, meta_feature=meta_feature)
    return model


def FluexNet_meta_geolater_old(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                  lambda_g4v=0.,
                  num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, meta_dim=1, means=4.05,
                  stds=1.14, meta_feature=True, **kwargs):
    model = Fluex_net_meta_geolater_old(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                           num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                           lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                           res=res, meta_dim=meta_dim, means=means, stds=stds, meta_feature=meta_feature)
    return model


def FluexNet_meta_twostep(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                  lambda_g4v=0.,
                  num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, meta_dim=1, means=4.05,
                  stds=1.14, meta_feature=True, **kwargs):
    model = Fluex_net_meta_TwoStep(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                           num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                           lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                           res=res, meta_dim=meta_dim, means=means, stds=stds, meta_feature=meta_feature)
    return model


def FluexNet_meta_omni(img_size=(48, 29), patch_size=(8, 9), in_chans=5, embed_dim=256, depth=2, latent_dim=256, num_heads=16, drop_path=0.0,
                  mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, lambda_g1v=1., lambda_g2v=0., lambda_g3v=0.,
                  lambda_g4v=0.,
                  num_sat=12, drop_sat=0., shell_num=2, num_res_blocks=1, rescale_loss=False, step_two=False, res=True, meta_dim=1, means=4.05,
                  stds=1.14, meta_feature=True, **kwargs):
    model = Fluex_net_meta_omni(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, latent_dim=latent_dim,
                           num_heads=num_heads, drop_path=drop_path, num_res_blocks=num_res_blocks, rescale_loss=rescale_loss,
                           mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer, lambda_g1v=lambda_g1v, lambda_g2v=lambda_g2v,
                           lambda_g3v=lambda_g3v, lambda_g4v=lambda_g4v, num_sat=num_sat, drop_sat=drop_sat, shell_num=shell_num, step_two=step_two,
                           res=res, meta_dim=meta_dim, means=means, stds=stds, meta_feature=meta_feature)
    return model

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count, flop_count_str, parameter_count

    torch.autograd.set_detect_anomaly(True)

    c = 6
    meta_dim = 3
    n = FluexNet_meta_future(img_size=(72, 29), step_two=True, in_chans=c, meta_dim=meta_dim, rescale_loss=False)
    x = torch.rand([2, 12, c, 72, 29])
    m = torch.randint(0, 2, (2, 12, c, 72, 29))
    y = torch.rand([2, 1, 72, 27])
    ym = torch.randint(0, 2, (2, 1, 72, 27))
    meta = torch.rand([2, 1, 72, 2 + meta_dim]) * 4
    metam = torch.randint(0, 2, (2, 1, 72, 2 + meta_dim))

    loss, pred = n(x, y, m, ym, meta, metam)
    print(loss.shape, pred.shape)

    loss.backward()

    params = parameter_count_table(n)
    print(params)

    # for pname, param in n.named_parameters():
    #     print(pname)
