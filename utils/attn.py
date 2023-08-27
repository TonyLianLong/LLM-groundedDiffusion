# visualization-related functions are in vis
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils

def get_token_attnv2(token_id, saved_attns, attn_key, attn_aggregation_step_start=10, input_ca_has_condition_only=False, return_np=False):
    """
    saved_attns: a list of saved_attn (list is across timesteps)
    
    moves to cpu by default
    """
    saved_attns = saved_attns[attn_aggregation_step_start:]    

    saved_attns = [saved_attn[attn_key].cpu() for saved_attn in saved_attns]
    
    attn = torch.stack(saved_attns, dim=0).mean(dim=0)
    
    # print("attn shape", attn.shape)
    
    # attn: (batch, head, spatial, text)

    if not input_ca_has_condition_only:
        assert attn.shape[0] == 2, f"Expect to have 2 items (uncond and cond), but found {attn.shape[0]} items"
        attn = attn[1]
    else:
        assert attn.shape[0] == 1, f"Expect to have 1 item (cond only), but found {attn.shape[0]} items"
        attn = attn[0]
    attn = attn.mean(dim=0)[:, token_id]
    H = W = int(math.sqrt(attn.shape[0]))
    attn = attn.reshape((H, W))
    
    if return_np:
        return attn.numpy()

    return attn

def shift_saved_attns_item(saved_attns_item, offset, guidance_attn_keys, horizontal_shift_only=False):
    """
    `horizontal_shift_only`: only shift horizontally. If you use `offset` from `compose_latents_with_alignment` with `horizontal_shift_only=True`, the `offset` already has y_offset = 0 and this option is not needed.
    """
    x_offset, y_offset = offset
    if horizontal_shift_only:
        y_offset = 0.
    
    new_saved_attns_item = {}
    for k in guidance_attn_keys:
        attn_map = saved_attns_item[k]
        
        attn_size = attn_map.shape[-2]
        attn_h = attn_w = int(math.sqrt(attn_size))
        # Example dimensions: [batch_size, num_heads, 8, 8, num_tokens]
        attn_map = attn_map.unflatten(2, (attn_h, attn_w))
        attn_map = utils.shift_tensor(
            attn_map, x_offset, y_offset, 
            offset_normalized=True, ignore_last_dim=True
        )
        attn_map = attn_map.flatten(2, 3)
        
        new_saved_attns_item[k] = attn_map
        
    return new_saved_attns_item

def shift_saved_attns(saved_attns, offset, guidance_attn_keys, **kwargs):
    # Iterate over timesteps
    shifted_saved_attns = [shift_saved_attns_item(saved_attns_item, offset, guidance_attn_keys, **kwargs) for saved_attns_item in saved_attns]
    
    return shifted_saved_attns


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
            
    Credit: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(
                    dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)
