"""Queue six point-counting baselines; collect best validation results."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WEIGHT = 'vgg16_bn-6c64b313.pth'
DATASETS = ('Wheat500', 'Rice1700')
MODELS = ('pet', 'sahcc', 'cgtipnet', 'p2rloss', 'p2pnet', 'cltr')
POINT_UPSTREAM = {
    'p2pnet': ('CrowdCounting-P2PNet', '5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455', 'models/p2pnet.py'),
    'cltr': ('CLTR', 'ab1b5e58eec4851cc629379abeeaceff908b48f4', 'Networks/CDETR/conditional_detr.py'),
}
SCORES = ('mae', 'rmse', 'f1_4px', 'f1_8px', 'f1_16px')
COLUMNS = ('dataset', 'model', 'seed', 'batch_size', 'status', 'best_epoch', *SCORES,
           'error', 'metrics_file')


def model_name(value):
    name = value.lower()
    return 'cgtipnet' if name == 'tipfuse' else name


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--repos-root', type=Path, default=ROOT.parent)
    p.add_argument('--weights', type=Path, default=ROOT/'pretrained'/WEIGHT)
    p.add_argument('--output-root', type=Path, default=ROOT/'outputs')
    p.add_argument('--run-name', default='baseline')
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    p.add_argument('--models', nargs='+', type=model_name, choices=MODELS, default=MODELS)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--pet-batch-size', type=int, default=8)
    p.add_argument('--sah-batch-size', type=int, default=40)
    p.add_argument('--cgtipnet-batch-size', '--tipfuse-batch-size', type=int, default=40)
    p.add_argument('--p2rloss-batch-size', type=int, default=16)
    p.add_argument('--p2pnet-batch-size', type=int, default=8)
    p.add_argument('--cltr-batch-size', type=int, default=4)
    p.add_argument('--cltr-queries', type=int, default=500)
    p.add_argument('--cltr-weights', type=Path, default=ROOT/'pretrained'/'resnet50-0676ba61.pth')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if min(a.epochs, a.eval_every, a.pet_batch_size, a.sah_batch_size, a.cgtipnet_batch_size, a.p2rloss_batch_size, a.p2pnet_batch_size, a.cltr_batch_size, a.cltr_queries) < 1 or a.workers < 0:
        p.error('epochs, validation frequency and batch sizes must be positive; workers >= 0')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', a.run_name):
        p.error('invalid run name')
    if len(set(a.datasets)) != len(a.datasets) or len(set(a.models)) != len(a.models):
        p.error('duplicate dataset or model')
    return a


def dataset_splits(root):
    for name in ('JPEGImages', 'PointAnnotations'):
        if not (root/name).is_dir():
            raise ValueError(f'Missing directory: {root/name}')
    ids, hashes = {}, {}
    for split in ('train', 'val'):
        path = root/'ImageSets'/f'{split}.txt'
        content = path.read_bytes()
        items = [line.strip() for line in content.decode('utf-8-sig').splitlines() if line.strip()]
        if not items or len(items) != len(set(items)):
            raise ValueError(f'Empty or duplicated IDs: {path}')
        ids[split] = set(items)
        hashes[split] = {'sha256': hashlib.sha256(content).hexdigest(), 'samples': len(items)}
    if ids['train'] & ids['val']:
        raise ValueError(f'Train/val IDs overlap: {root}')
    return hashes


def plan(a, dataset, model):
    if model in POINT_UPSTREAM:
        repo = (a.repos_root / POINT_UPSTREAM[model][0]).resolve()
        data = (a.data_root/dataset).resolve()
        run = (a.output_root/a.run_name/dataset/f'{model}_seed{a.seed}').resolve()
        dest = run/'artifacts'
        batch = a.p2pnet_batch_size if model == 'p2pnet' else a.cltr_batch_size
        weights = (a.weights if model == 'p2pnet' else a.cltr_weights).resolve()
        args = [sys.executable, '-u', str(ROOT/'train_point_baseline.py'),
                '--model', model, '--upstream', str(repo),
                '--data-code', str((a.repos_root/'tipnova').resolve()),
                '--data-path', str(data), '--weights', str(weights), '--output', str(dest),
                '--batch-size', str(batch), '--epochs', str(a.epochs),
                '--workers', str(a.workers), '--eval-every', str(a.eval_every),
                '--seed', str(a.seed)]
        if model == 'cltr':
            args += ['--cltr-queries', str(a.cltr_queries)]
        return repo, data, run, dest, batch, args, weights
    names = {'pet': 'PET', 'sahcc': 'sah-leaf-counting', 'cgtipnet': 'tipnova', 'p2rloss': 'p2rloss'}
    repo = (a.repos_root/names[model]).resolve()
    data = (a.data_root/dataset).resolve()
    run = (a.output_root/a.run_name/dataset/f'{model}_seed{a.seed}').resolve()
    batch = {'pet': a.pet_batch_size, 'sahcc': a.sah_batch_size, 'cgtipnet': a.cgtipnet_batch_size,
             'p2rloss': a.p2rloss_batch_size}[model]
    args = [sys.executable, '-u']
    if model == 'pet':
        dest = run/'artifacts'
        args += ['main.py', '--dataset_file', 'RiceW3999', '--data_path', str(data),
                 '--batch_size', str(batch), '--epochs', str(a.epochs),
                 '--num_workers', str(a.workers), '--eval_freq', str(a.eval_every),
                 '--seed', str(a.seed), '--output_dir', str(dest), '--auto-resume']
        weights = repo/'pretrained'/WEIGHT
    elif model == 'p2rloss':
        dest = run/'train'/'output'
        weights = a.weights.resolve()
        # P2RLoss derives OUTPUT with os.path.join('exp', TAG, 'output').
        # An absolute tag keeps artifacts inside this benchmark job.
        args += ['main.py', '--data-path', str(data), '--label', '100',
                 '--batch-size', str(batch), '--tag', str(run/'train'), '--opts',
                 'DATA.DATASET', 'ricew3999', 'DATA.NUM_WORKERS', str(a.workers),
                 'TRAIN.EPOCHS', str(a.epochs), 'SAVE_FREQ', str(a.eval_every),
                 'SEED', str(a.seed), 'TRAIN.AUTO_RESUME', 'True',
                 'MODEL.PRETRAINED', str(weights), 'TEST.LOC_RADII', '[4.0, 8.0, 16.0]']
        return repo, data, run, dest, batch, args, weights
    else:
        dest = run/'train'/'output'
        args += ['train.py', '--data-path', str(data), '--pretrained', str(a.weights.resolve()),
                 '--batch-size', str(batch), '--epochs', str(a.epochs),
                 '--workers', str(a.workers), '--eval-every', str(a.eval_every),
                 '--seed', str(a.seed), '--output-root', str(run), '--tag', 'train', '--auto-resume']
        if model == 'cgtipnet':
            args += ['--variant', 'cgtipnet']
        weights = a.weights.resolve()
    args += ['--device', 'cuda', '--loc-radii', '4', '8', '16']
    return repo, data, run, dest, batch, args, weights


def atomic_write(path, fn):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        fn(stream)
    os.replace(temp, path)


def save_json(path, obj):
    atomic_write(path, lambda stream: json.dump(obj, stream, indent=2, ensure_ascii=False))


def load_result(path, expected_samples):
    record = json.loads(path.read_text(encoding='utf-8'))
    metrics = record['metrics']
    if metrics.get('samples') != expected_samples or record.get('protocol', {}).get('radii') != [4, 8, 16]:
        raise ValueError(f'Unexpected validation sample count or matching radii: {path}')
    scores = {key: float(metrics[key]) for key in SCORES}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError('Non-finite validation metrics')
    epoch = record.get('metadata', {}).get('epoch_index')
    return {**scores, 'best_epoch': '' if epoch is None else int(epoch)+1}


def check_repo(repo, args, model):
    entry = repo/(POINT_UPSTREAM[model][2] if model in POINT_UPSTREAM else args[2])
    if not entry.is_file():
        raise ValueError(f'Missing trainer: {entry}')
    if model == 'cgtipnet':
        source = entry.read_text(encoding='utf-8')
        if '--variant' not in source or "'cgtipnet'" not in source:
            raise ValueError(f'{repo}: update tipnova/main to a version supporting --variant cgtipnet')
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True)
    if status.strip():
        raise ValueError(f'Commit or stash tracked source changes before launching: {repo}')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()


def main(argv=None):
    a = parse_args(argv)
    jobs = []
    for dataset, model in itertools.product(a.datasets, a.models):
        repo, data, run, dest, batch, args, weights = plan(a, dataset, model)
        if a.dry_run:
            print(f'{dataset}/{model} [batch={batch}] | cwd={repo}')
            print(shlex.join(args))
            continue
        split_hash = dataset_splits(data)
        revision = check_repo(repo, args, model)
        extra = {}
        if model in POINT_UPSTREAM:
            expected = POINT_UPSTREAM[model][1]
            if revision != expected:
                raise ValueError(f'{repo}: expected upstream revision {expected}; see README')
            data_repo = (a.repos_root/'tipnova').resolve()
            extra['data_code_revision'] = check_repo(data_repo, ['', '', 'dataset.py'], 'data')
            extra['adapter_sha256'] = hashlib.sha256((ROOT/'train_point_baseline.py').read_bytes()).hexdigest()
        if not weights.is_file():
            raise ValueError(f'Missing pretrained weight: {weights}')
        st = weights.stat()
        manifest = dict(command=args, repo=str(repo), revision=revision, splits=split_hash,
                        weight_path=str(weights), weight_size=st.st_size, weight_mtime_ns=st.st_mtime_ns, **extra)
        config = run/'config.json'
        if model == 'cgtipnet' and not run.exists() and run.with_name(f'tipfuse_seed{a.seed}').exists():
            raise ValueError('Legacy TipFuse run exists. Use the old benchmark revision to resume it, '
                             'or use a new --run-name for CGTipNet.')
        if run.exists():
            if not a.resume:
                raise ValueError(f'Run exists: {run}. Use --resume or a new --run-name.')
            if not config.is_file() or json.loads(config.read_text(encoding='utf-8')) != manifest:
                raise ValueError(f'Resume configuration changed: {run}')
        jobs.append((dataset, model, repo, run, dest, batch, args, manifest))
    if a.dry_run:
        return 0
    summary = (a.output_root/a.run_name/'results.csv').resolve()
    summary.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if summary.exists():
        if not a.resume:
            raise ValueError(f'Suite already exists: {summary}. Use --resume or a new --run-name.')
        with summary.open(encoding='utf-8', newline='') as stream:
            rows.update({(r['dataset'], r['model'], int(r['seed'])): r for r in csv.DictReader(stream)})
    failed = False
    for dataset, model, repo, run, dest, batch, args, manifest in jobs:
        run.mkdir(parents=True, exist_ok=True)
        save_json(run/'config.json', manifest)
        best = dest/'best_metrics.json'
        marker = run/'completed.json'
        row = dict(dataset=dataset, model=model, seed=a.seed, batch_size=batch, metrics_file=str(best))
        print(f'\n[{dataset} / {model}] batch={batch} | {run}', flush=True)
        try:
            if not marker.is_file():
                with (run/'console.log').open('a', encoding='utf-8') as log:
                    with subprocess.Popen(args, cwd=repo, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, text=True, errors='replace') as child:
                        try:
                            for line in child.stdout:
                                print(line, end='', flush=True)
                                log.write(line)
                                log.flush()
                        except KeyboardInterrupt:
                            child.terminate()
                            raise
                        if child.wait():
                            raise RuntimeError(f'Trainer exited {child.returncode}; see console.log')
                row.update(load_result(best, manifest['splits']['val']['samples']))
                save_json(marker, {'epochs': a.epochs})
            else:
                row.update(load_result(best, manifest['splits']['val']['samples']))
                print('Already completed; skipped.')
            row['status'] = 'ok'
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            failed = True
            row.update(status='failed', error=str(exc))
            print(f'FAILED: {exc}', file=sys.stderr, flush=True)
        rows[(dataset, model, a.seed)] = row
        def write(stream):
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows.values())
        atomic_write(summary, write)
    print(f'Validation results: {summary}')
    return int(failed)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
