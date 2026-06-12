import torch

def pad_tensor(tensor, dim1, dim2, mode = 'constant', value=0):
    # Check if the first two dimensions are not the same
    if tensor[dim1] != tensor[dim2]:
        # Calculate the padding required for each side
        pad_amount = abs(tensor[dim1] - tensor[dim2]) // 2
        # Pad the tensor along the shorter dimension
        if tensor[dim1] < tensor[dim2]:
            padding = (pad_amount, pad_amount, 0, 0, 0, 0, 0, 0)
        else:
            padding = (0, 0, pad_amount, pad_amount, 0, 0, 0, 0)
        # Pad the tensor with zeros
        padded_tensor = torch.nn.functional.pad(tensor, padding, mode=mode, value=value)
        return padded_tensor
    else:
        print("First two dimensions are already the same.")
        return tensor