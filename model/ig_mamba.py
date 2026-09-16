import warnings
import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    _MAMBA_SSM_AVAILABLE = True
except ImportError:
    _MAMBA_SSM_AVAILABLE = False


class IGMambaBlock(nn.Module):
    """
    VI-gated block for the IG-Mamba encoder/decoder stages.

    Two modes:
      - Real mode (mamba_ssm installed): the VI gate modulates features,
        THEN a real Mamba selective-scan mixes information across the
        flattened spatial sequence (token-to-token interaction). This is
        what makes it an actual "Mamba" block -- sequence mixing, not just
        a per-pixel gate.
      - Fallback mode (mamba_ssm NOT installed): identical to the original
        placeholder -- LayerNorm + VI-gated sigmoid, no token mixing at all.
        Every position is processed independently. This was the ONLY mode
        available before this change, and is silently what "IGMambaBlock"
        meant despite the name -- see identify_class3.py-adjacent audit
        notes: there was no actual selective scan happening anywhere in the
        forward pass.

    mamba_ssm requires causal-conv1d compiled against CUDA, which is
    Linux-oriented and typically won't install cleanly on Windows dev boxes.
    That's why this falls back gracefully instead of hard-requiring it --
    write/debug on Windows with the placeholder, then get the real block for
    free the moment you `pip install mamba-ssm causal-conv1d` on Linux/cloud
    without any code changes needed here.
    """

    _warned_once = False

    def __init__(self, d_model, use_real_mamba=True, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.vi_proj = nn.Linear(1, d_model)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        self.norm = nn.LayerNorm(d_model)

        self.use_real_mamba = use_real_mamba and _MAMBA_SSM_AVAILABLE

        if self.use_real_mamba:
            self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        elif use_real_mamba and not _MAMBA_SSM_AVAILABLE and not IGMambaBlock._warned_once:
            warnings.warn(
                "mamba_ssm is not installed -- IGMambaBlock is running in "
                "PLACEHOLDER mode (VI gate only, no real sequence mixing/scan). "
                "This is fine for CPU/Windows dev and smoke tests, but results "
                "from this mode should not be reported as 'Mamba-based' in a "
                "paper. Install mamba_ssm + causal-conv1d on Linux/cloud (GPU "
                "required) to get the real selective-scan kernel.",
                stacklevel=2,
            )
            IGMambaBlock._warned_once = True

    def forward(self, x, vi_mask):
        # x: [Batch, Seq, d_model]
        # vi_mask: [Batch, Seq, 1]

        # Physics-informed gate: filter information flow by vegetation index.
        vi_gate = torch.sigmoid(self.vi_proj(vi_mask) * self.gamma)
        x = self.norm(x) * vi_gate

        if self.use_real_mamba:
            # Real selective-scan mixing across the sequence dimension --
            # this is the step that was previously entirely missing.
            # Residual connection so the scan refines the gated features
            # rather than replacing them outright.
            x = x + self.mamba(x)

        return x