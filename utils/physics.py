import torch
import torch.nn.functional as F


def compute_ndvi(nir, red, eps=1e-8):
    """
    Normalized Difference Vegetation Index: NDVI = (NIR - Red) / (NIR + Red)

    nir, red: [H, W] tensors (single band each)
    Returns: [H, W] tensor, values roughly in [-1, 1]. Higher = more vegetation.
    """
    return (nir - red) / (nir + red + eps)


def compute_exg(red, green, blue):
    """
    Excess Green Index: ExG = 2*G - R - B

    A fallback vegetation index that only needs RGB (no NIR band required).
    Useful if a sensor/dataset doesn't provide a NIR channel.
    """
    return 2 * green - red - blue


def build_vi_pyramid(vi, pool_factors=(4, 8, 16, 32)):
    """
    Downsamples a single-channel VI mask into the multi-scale pyramid that
    gates each IG-Mamba encoder stage.

    vi: [1, H, W] tensor (channel dim already unsqueezed)
    pool_factors: downsampling factor for each pyramid level, applied to the
        ORIGINAL resolution (not cumulatively). For a 512x512 input, the stem
        (stride-4 patchify) plus three further stride-2 merges produce stage
        resolutions of 128/64/32/16 -- so the matching pool factors from 512
        are 4, 8, 16, 32.

        If you change the input resolution or the stem/merge strides, these
        factors must be recomputed to match, or you'll hit the same
        shape-mismatch bug this pipeline had before ("size of tensor a must
        match tensor b at non-singleton dimension").

    Returns: dict {'vi_1': [1,H/4,W/4], 'vi_2': [1,H/8,W/8],
                   'vi_3': [1,H/16,W/16], 'vi_4': [1,H/32,W/32]}
    """
    return {
        f'vi_{i + 1}': F.avg_pool2d(vi, factor)
        for i, factor in enumerate(pool_factors)
    }


def extract_vi_pyramid(nir, red, pool_factors=(4, 8, 16, 32), eps=1e-8):
    """
    Convenience wrapper matching the diagram's "VI extractor -> VI pyramid"
    flow in one call: computes NDVI from raw bands, then builds the pyramid.

    nir, red: [H, W] tensors
    Returns: dict of vi_1..vi_4, same as build_vi_pyramid.
    """
    ndvi = compute_ndvi(nir, red, eps=eps).unsqueeze(0)  # [1, H, W]
    return build_vi_pyramid(ndvi, pool_factors=pool_factors)