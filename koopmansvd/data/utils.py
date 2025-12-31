import torch


def extract_raw_data(x, batch_size):
    """Helper to extract raw data numpy arrays from batch dict or tensor."""
    if isinstance(x, dict):
        # For molecular datasets (SchNet)
        if "_positions" in x:
            pos = x["_positions"]
        elif "R" in x:
            pos = x["R"]
        else:
            return None

        if pos is not None:
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu()
            return pos.reshape(batch_size, -1).numpy()
    else:
        # For tensor datasets
        x_np = x.cpu().numpy()
        if x_np.ndim > 2:
            return x_np.reshape(batch_size, -1)
        return x_np
    return None
