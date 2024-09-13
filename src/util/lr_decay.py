def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.seismic_encoder.blocks) + len(model.velocity_decoder.decoder_blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        nwd = False

        for l in no_weight_decay_list:
            if l in n:
                nwd = True
                break

        if p.ndim == 1 or nwd:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers, int(len(model.seismic_encoder.blocks)))
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers, en_layer):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if 'cls_token' in name or 'pos_embed' in name:
        return 0
    elif 'patch_embed' in name or 'mask_token' in name:
        return 0
    elif 'neck' in name or 'trans' in name:
        return num_layers
    elif 'decoder_pred' in name or 'cnn' in name:
        return num_layers
    elif 'seismic_encoder.blocks' in name:
        return int(name.split('.')[2]) + 1
    elif 'velocity_decoder.decoder_blocks' in name:
        return int(name.split('.')[2]) + 1 + en_layer
    else:
        return num_layers