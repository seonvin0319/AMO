"""Prepare/launch 12 direct-Q runs. Default is a read-only command preview."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--phase', choices=['train', 'eval'], default='train')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output-root', type=Path, default=ROOT/'results/directq_rms_a5')
    p.add_argument('--run', help='Optional exact run ID (for independent workers)')
    args = p.parse_args()
    spec = ROOT/'experiments/directq_rms'
    manifest = json.loads((spec/'manifest.json').read_text())
    assert len(manifest) == 12 and len({r['id'] for r in manifest}) == 12
    hashes = json.loads((spec/'runtime_hashes.json').read_text())
    for path, expected in hashes['git_blob_sha1'].items():
        raw = (ROOT/path).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if actual != expected:
            p.error(f'Runtime differs from prepared source: {path}')
    runs = [r for r in manifest if not args.run or r['id'] == args.run]
    if not runs:
        p.error('Unknown run ID')
    output = args.output_root.resolve()
    commands = []
    for r in runs:
        config = load_config('td3_amo', r['env'], ROOT/r['config'])
        tag = 'r2e3' if r['alpha_lr'] == .002 else 'r3e4'
        base = load_config('td3_amo', r['env'], ROOT/f'configs/td3_amo_bootrms_a5_{tag}.yaml')
        # Archived 3-task controls used 10x5 final evaluation, including AntMaze.
        base['eval_episodes'] = 10
        diff = {k for k in config if config[k] != base[k]}
        if diff != {'execution_score'} or config['execution_score'] != 'direct_q':
            p.error(f'Unexpected baseline differences: {r["id"]}: {diff}')
        if args.phase == 'train':
            cmd = [sys.executable, str(ROOT/'train.py'), '--algorithm', 'td3_amo',
                   '--backend', 'jax', '--env', r['env'], '--config', str(ROOT/r['config']),
                   '--seed', str(r['seed']), '--device', args.device,
                   '--output', str(output/r['id']), '--log-every', '5000',
                   '--save-every', '200000', '--no-eval']
        else:
            cmd = [sys.executable, str(ROOT/'scripts/eval_checkpoints_cpu.py'),
                   '--runs-root', str(output), '--run', r['id'], '--final-only',
                   '--episodes', '10', '--final-repeats', '5', '--device', 'cpu']
        commands.append((r,cmd))
    # Validate all selected destinations before starting the first process.
    if args.execute:
        for r, _ in commands:
            dest = output/r['id']
            if args.phase == 'train' and dest.exists():
                p.error(f'Run directory already exists: {dest}; use --run for remaining jobs')
            if args.phase == 'eval' and not (dest/'checkpoints/step_1000000.npz').is_file():
                p.error(f'Missing final checkpoint: {dest}')
    for r, cmd in commands:
        print(shlex.join(cmd), flush=True)
        if not args.execute:
            continue
        env = dict(os.environ)
        if args.phase == 'eval':
            env.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
        else:
            env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
            env.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0')
            dest = output/r['id']
            dest.mkdir(parents=True, exist_ok=False)
            # train.py requires an empty directory; provenance is stored beside it.
            (output/(r['id']+'.launch.json')).write_text(json.dumps({
                'run': r, 'command': cmd, 'runtime': hashes,
                'config_sha256': hashlib.sha256((ROOT/r['config']).read_bytes()).hexdigest(),
            }, indent=2)+'\n')
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        if args.phase == 'eval':
            payload = json.loads((output/r['id']/'posthoc_eval_cpu/final.json').read_text())
            assert payload['ok'] and payload['row']['step'] == 1000000
            assert payload['row']['episodes'] == 50


if __name__ == '__main__':
    main()
