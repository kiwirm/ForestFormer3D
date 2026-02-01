import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.in_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        key = 'state_dict'
    elif 'model' in checkpoint:
        key = 'model'
    else:
        key = None

    state_dict = checkpoint if key is None else checkpoint[key]
    for layer, weight in list(state_dict.items()):
        layer_name = layer
        if layer_name.startswith('model.'):
            layer_name = layer_name[len('model.'):]
        if not (layer_name.startswith('unet') or layer_name.startswith('input_conv')):
            continue
        if not layer_name.endswith('weight'):
            continue
        if not hasattr(weight, 'shape') or len(weight.shape) != 5:
            continue
        # Only permute if it's still in (k,k,k,in,out) order.
        if weight.shape[0] in (1, 2, 3) and weight.shape[-1] not in (1, 2, 3):
            # (k,k,k,in,out) -> (out,k,k,k,in)
            state_dict[layer] = weight.permute(4, 0, 1, 2, 3)

    if key is not None:
        checkpoint[key] = state_dict
    torch.save(checkpoint, args.out_path)
