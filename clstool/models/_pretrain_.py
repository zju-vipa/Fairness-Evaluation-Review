# local paths (high priority)
model_local_paths = {
    'your_model_1': '/local/path/to/the/pretrained',
    'your_model_2': ['/local/path_1/to/the/pretrained', '/local/path_2/to/the/pretrained'],
}

# urls (low priority)
model_urls = {
    'your_model': 'url://to/the/pretrained',

    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',

    'cait_xxs24_224': 'https://dl.fbaipublicfiles.com/deit/XXS24_224.pth',
    'cait_xxs24_384': 'https://dl.fbaipublicfiles.com/deit/XXS24_384.pth',
    'cait_xxs36_224': 'https://dl.fbaipublicfiles.com/deit/XXS36_224.pth',
    'cait_xxs36_384': 'https://dl.fbaipublicfiles.com/deit/XXS36_384.pth',
    'cait_xs24_384': 'https://dl.fbaipublicfiles.com/deit/XS24_384.pth',
    'cait_s24_224': 'https://dl.fbaipublicfiles.com/deit/S24_224.pth',
    'cait_s24_384': 'https://dl.fbaipublicfiles.com/deit/S24_384.pth',
    'cait_s36_384': 'https://dl.fbaipublicfiles.com/deit/S36_384.pth',
    'cait_m36_384': 'https://dl.fbaipublicfiles.com/deit/M36_384.pth',
    'cait_m48_448': 'https://dl.fbaipublicfiles.com/deit/M48_448.pth',

    'convnext_tiny': 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth',
    'convnext_small': 'https://download.pytorch.org/models/convnext_small-0c510722.pth',
    'convnext_base': 'https://download.pytorch.org/models/convnext_base-6075fbad.pth',
    'convnext_large': 'https://download.pytorch.org/models/convnext_large-ea097f82.pth',

    'deit_tiny_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
    'deit_small_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
    'deit_base_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
    'deit_base_patch16_384': 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
    'deit_tiny_distilled_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
    'deit_small_distilled_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
    'deit_base_distilled_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
    'deit_base_distilled_patch16_384': 'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
    'deit3_small_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_3_small_224_1k.pth',
    'deit3_small_patch16_384': 'https://dl.fbaipublicfiles.com/deit/deit_3_small_384_1k.pth',
    'deit3_medium_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_1k.pth',
    'deit3_base_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_3_base_224_1k.pth',
    'deit3_base_patch16_384': 'https://dl.fbaipublicfiles.com/deit/deit_3_base_384_1k.pth',
    'deit3_large_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_3_large_224_1k.pth',
    'deit3_large_patch16_384': 'https://dl.fbaipublicfiles.com/deit/deit_3_large_384_1k.pth',
    'deit3_huge_patch14_224': 'https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_1k.pth',
    'deit3_small_patch16_224_in21ft1k': 'https://dl.fbaipublicfiles.com/deit/deit_3_small_224_21k.pth',
    'deit3_small_patch16_384_in21ft1k': 'https://dl.fbaipublicfiles.com/deit/deit_3_small_384_21k.pth',
    'deit3_medium_patch16_224_in21ft1k': 'https://dl.fbaipublicfiles.com/deit/deit_3_medium_224_21k.pth',
    'deit3_base_patch16_224_in21ft1k': 'https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth',
    'deit3_base_patch16_384_in21ft1k': 'https://dl.fbaipublicfiles.com/deit/deit_3_base_384_21k.pth',
    'deit3_large_patch16_224_in21ft1k': 'https://dl.fbaipublicfiles.com/deit/deit_3_large_224_21k.pth',
    'deit3_large_patch16_384_in21ft1k': 'https://dl.fbaipublicfiles.com/deit/deit_3_large_384_21k.pth',
    'deit3_huge_patch14_224_in21ft1k': 'https://dl.fbaipublicfiles.com/deit/deit_3_huge_224_21k_v1.pth',

    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',

    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    'efficientnet_b0': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth',
    'efficientnet_b1': 'https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth',
    'efficientnet_b2': 'https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth',
    'efficientnet_b3': 'https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth',
    'efficientnet_b4': 'https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth',
    # Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
    'efficientnet_b5': 'https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth',
    'efficientnet_b6': 'https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth',
    'efficientnet_b7': 'https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth',

    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',

    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',

    'levit_128s': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128S-96703c44.pth',
    'levit_128': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-128-b88c2750.pth',
    'levit_192': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-192-92712e41.pth',
    'levit_256': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-256-13b5763e.pth',
    'levit_256d': None,
    'levit_384': 'https://dl.fbaipublicfiles.com/LeViT/LeViT-384-9bdaf2e2.pth',

    'mixer_s32_224': None,
    'mixer_s16_224': None,
    'mixer_b32_224': None,
    'mixer_b16_224': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth',
    'mixer_b16_224_in21k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224_in21k-617b3de2.pth',
    'mixer_l32_224': None,
    'mixer_l16_224': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth',
    'mixer_l16_224_in21k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224_in21k-846aa33c.pth',
    'mixer_b16_224_miil_in21k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mixer_b16_224_miil_in21k-2a558a71.pth',
    'mixer_b16_224_miil': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mixer_b16_224_miil-9229a591.pth',
    'gmixer_12_224': None,
    'gmixer_24_224': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmixer_24_224_raa-7daf7ae6.pth',
    'resmlp_12_224': 'https://dl.fbaipublicfiles.com/deit/resmlp_12_no_dist.pth',
    'resmlp_24_224': 'https://dl.fbaipublicfiles.com/deit/resmlp_24_no_dist.pth',
    'resmlp_36_224': 'https://dl.fbaipublicfiles.com/deit/resmlp_36_no_dist.pth',
    'resmlp_big_24_224': 'https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth',
    'resmlp_12_distilled_224': 'https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth',
    'resmlp_24_distilled_224': 'https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth',
    'resmlp_36_distilled_224': 'https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth',
    'resmlp_big_24_distilled_224': 'https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth',
    'resmlp_big_24_224_in22ft1k': 'https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth',
    'resmlp_12_224_dino': 'https://dl.fbaipublicfiles.com/deit/resmlp_12_dino.pth',
    'resmlp_24_224_dino': 'https://dl.fbaipublicfiles.com/deit/resmlp_24_dino.pth',
    'gmlp_ti16_224': None,
    'gmlp_s16_224': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmlp_s16_224_raa-10536d42.pth',
    'gmlp_b16_224': None,

    'mnasnet0_5': 'https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth',
    'mnasnet0_75': None,
    'mnasnet1_0': 'https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth',
    'mnasnet1_3': None,

    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',

    'mobilenet_v3_large': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
    'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',

    'poolformer_s12': 'https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar',
    'poolformer_s24': 'https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar',
    'poolformer_s36': 'https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar',
    'poolformer_m36': 'https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar',
    'poolformer_m48': 'https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar',

    'pvt_tiny': None,
    'pvt_small': None,
    'pvt_medium': None,
    'pvt_large': None,
    'pvt_huge_v2': None,

    'regnet_y_400mf': 'https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth',
    'regnet_y_800mf': 'https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth',
    'regnet_y_1_6gf': 'https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth',
    'regnet_y_3_2gf': 'https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth',
    'regnet_y_8gf': 'https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth',
    'regnet_y_16gf': 'https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth',
    'regnet_y_32gf': 'https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth',
    'regnet_x_400mf': 'https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth',
    'regnet_x_800mf': 'https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth',
    'regnet_x_1_6gf': 'https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth',
    'regnet_x_3_2gf': 'https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth',
    'regnet_x_8gf': 'https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth',
    'regnet_x_16gf': 'https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth',
    'regnet_x_32gf': 'https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth',

    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',

    'shufflenet_v2_x0_5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenet_v2_x1_0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenet_v2_x1_5': None,
    'shufflenet_v2_x2_0': None,

    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth',

    'swin_tiny_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    'swin_small_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
    'swin_base_patch4_window12_384': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth',
    'swin_base_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',
    'swin_large_patch4_window12_384': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth',
    'swin_large_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth',
    'swin_base_patch4_window12_384_in22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
    'swin_base_patch4_window7_224_in22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
    'swin_large_patch4_window12_384_in22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
    'swin_large_patch4_window7_224_in22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',

    'swinv2_tiny_window8_256': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth',
    'swinv2_tiny_window16_256': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window16_256.pth',
    'swinv2_small_window8_256': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window8_256.pth',
    'swinv2_small_window16_256': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window16_256.pth',
    'swinv2_base_window8_256': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth',
    'swinv2_base_window16_256': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window16_256.pth',
    'swinv2_base_window12_192_22k': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth',
    'swinv2_base_window12to16_192to256_22kft1k': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth',
    'swinv2_base_window12to24_192to384_22kft1k': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth',
    'swinv2_large_window12_192_22k': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12_192_22k.pth',
    'swinv2_large_window12to16_192to256_22kft1k': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to16_192to256_22kto1k_ft.pth',
    'swinv2_large_window12to24_192to384_22kft1k': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_large_patch4_window12to24_192to384_22kto1k_ft.pth',

    'tnt_s_patch16_224': 'https://github.com/contrastive/pytorch-image-models/releases/download/TNT/tnt_s_patch16_224.pth.tar',
    'tnt_b_patch16_224': None,

    'twins_pcpvt_small': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_pcpvt_small-e70e7e7a.pth',
    'twins_pcpvt_base': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_pcpvt_base-e5ecb09b.pth',
    'twins_pcpvt_large': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_pcpvt_large-d273f802.pth',
    'twins_svt_small': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_small-42e5f78c.pth',
    'twins_svt_base': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_base-c2265010.pth',
    'twins_svt_large': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_large-90f6aaa9.pth',

    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',

    'vit_b_16': 'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
    'vit_b_32': 'https://download.pytorch.org/models/vit_b_32-d86f8d99.pth',
    'vit_l_16': 'https://download.pytorch.org/models/vit_l_16-852ce7e3.pth',
    'vit_l_32': 'https://download.pytorch.org/models/vit_l_32-c7638314.pth',

    'vit_tiny_patch4_32': None,
    'vit_tiny_patch16_224': None,
    'vit_tiny_patch16_384': None,
    'vit_small_patch32_224': None,
    'vit_small_patch32_384': None,
    'vit_small_patch16_224': None,
    'vit_small_patch16_384': None,
    'vit_small_patch8_224': None,
    'vit_base_patch32_224': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
    'vit_base_patch32_384': None,
    'vit_base_patch16_224': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
    'vit_base_patch16_384': None,
    'vit_base_patch8_224': None,
    'vit_large_patch32_224': 'https://github.com/rwightman/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
    'vit_large_patch32_384': None,
    'vit_large_patch16_224': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
    'vit_large_patch16_384': None,
    'vit_large_patch14_224': None,
    'vit_huge_patch14_224': None,
    'vit_giant_patch14_224': None,
}
