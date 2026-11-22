"""Exercise the pinned LeRobot ACT forward and optimizer path on CUDA."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, help='Optional replay-derived NPZ batch')
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device('cuda')
    cameras = ('observation.images.image', 'observation.images.wrist_image')
    inputs = {key: PolicyFeature(FeatureType.VISUAL, (3, 64, 64)) for key in cameras}
    inputs['observation.state'] = PolicyFeature(FeatureType.STATE, (8,))
    config = ACTConfig(
        input_features=inputs,
        output_features={'action': PolicyFeature(FeatureType.ACTION, (7,))},
        device='cuda', pretrained_backbone_weights=None, push_to_hub=False,
        chunk_size=4, n_action_steps=4, dim_model=64, n_heads=4, dim_feedforward=128,
        n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1,
    )
    torch.cuda.reset_peak_memory_stats()
    policy = ACTPolicy(config).to(device)
    if args.batch:
        with np.load(args.batch, allow_pickle=False) as samples:
            batch = {name: torch.as_tensor(samples[name], device=device) for name in (*inputs, 'action')}
    else:
        batch = {key: torch.rand((2, 3, 64, 64), device=device) for key in cameras}
        batch['observation.state'] = torch.zeros((2, 8), device=device)
        batch['action'] = torch.zeros((2, 4, 7), device=device)
    policy.eval()
    with torch.no_grad():
        chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (2, 4, 7) and torch.isfinite(chunk).all()

    batch['action_is_pad'] = torch.zeros((2, 4), dtype=torch.bool, device=device)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    changed = next(p for p in policy.parameters() if p.grad is not None and p.grad.abs().max() > 0)
    before = changed.detach().clone()
    optimizer.step()
    assert not torch.equal(before, changed.detach())
    torch.cuda.synchronize()
    peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)

    policy.eval()
    with torch.no_grad(), TemporaryDirectory(prefix='act-cuda-smoke-') as temp_dir:
        expected = policy.predict_action_chunk(batch)
        checkpoint = Path(temp_dir) / 'act.pt'
        torch.save(policy.state_dict(), checkpoint)
        changed.zero_()
        policy.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        torch.testing.assert_close(policy.predict_action_chunk(batch), expected, rtol=0, atol=0)

    print(json.dumps({
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(), 'seed': 0, 'batch': 'replay' if args.batch else 'synthetic',
        'chunk_shape': list(chunk.shape), 'loss': loss.item(),
        'parameter_update': True, 'checkpoint_reload': True,
        'peak_allocated_mib': peak_allocated_mib,
    }, indent=2))


if __name__ == '__main__':
    main()
