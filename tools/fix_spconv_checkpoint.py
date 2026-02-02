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
    kernel_sizes = {1, 2, 3, 5, 7}
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

        shape = list(weight.shape)
        kernel_idx = [i for i, dim in enumerate(shape) if dim in kernel_sizes]
        if len(kernel_idx) not in (3, 4):
            continue

        if kernel_idx == [1, 2, 3]:
            # Already (out, k, k, k, in)
            continue
        if kernel_idx == [0, 1, 2]:
            # (k, k, k, in, out) -> (out, k, k, k, in)
            state_dict[layer] = weight.permute(4, 0, 1, 2, 3)
            continue
        if kernel_idx == [0, 1, 2, 3]:
            # (k, k, k, in, out) with in also == k (e.g., input_conv) -> (out, k, k, k, in)
            state_dict[layer] = weight.permute(4, 0, 1, 2, 3)
            continue
        if kernel_idx == [2, 3, 4]:
            # (out, in, k, k, k) or (in, out, k, k, k) -> (out, k, k, k, in)
            dim0_small = shape[0] in kernel_sizes
            dim1_small = shape[1] in kernel_sizes
            if dim0_small and not dim1_small:
                # (in, out, k, k, k)
                state_dict[layer] = weight.permute(1, 2, 3, 4, 0)
            else:
                # (out, in, k, k, k) or ambiguous -> assume out,in
                state_dict[layer] = weight.permute(0, 2, 3, 4, 1)

    if key is not None:
        checkpoint[key] = state_dict
    torch.save(checkpoint, args.out_path)
