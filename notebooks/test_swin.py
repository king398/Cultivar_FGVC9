import Swin
from torch import nn
import torch


def swinv2_base_patch4_window12to16_192to256_22kto1k_ft(pretrained=False, **kwargs):
    model = Swin.models.swin_transformer_v2.SwinTransformerV2(
        img_size=256,
        patch_size=4,
        in_chans=3, num_classes=1000,
        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        window_size=16, mlp_ratio=4., qkv_bias=True,
        drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2,
        norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
        use_checkpoint=False, pretrained_window_sizes=[12, 12, 12, 6]

    )
    model = model.load_state_dict(torch.load(
        '/home/mithil/PycharmProjects/Cultivar_FGVC9/models/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth'),
        strict=False)

