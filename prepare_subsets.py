"""Create lightweight Wheat500/Rice1700 views from existing RiceW3999 splits."""
import argparse
from pathlib import Path

from batch_train import ROOT, dataset_splits

SOURCES = {'Wheat500': ('wheat500__', 500), 'Rice1700': ('ricetip__', 1700)}


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    dataset_splits(source)
    splits = {
        split: [line.strip() for line in (source/'ImageSets'/f'{split}.txt').read_text(
            encoding='utf-8-sig').splitlines() if line.strip()]
        for split in ('train', 'val')
    }
    plans = {}
    for name, (prefix, expected) in SOURCES.items():
        target = output/name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Already exists: {target}; existing datasets are not overwritten')
        plan = {split: [i for i in ids if i.startswith(prefix)] for split, ids in splits.items()}
        if any(not ids for ids in plan.values()) or sum(map(len, plan.values())) != expected:
            raise ValueError(f'{name}: expected {expected} items across train and val; got '
                             f'{ {key: len(ids) for key, ids in plan.items()} }')
        plans[name] = plan
    for name, plan in plans.items():
        target = output/name
        image_sets = target/'ImageSets'
        image_sets.mkdir(parents=True)
        for folder in ('JPEGImages', 'PointAnnotations'):
            (target/folder).symlink_to(source/folder, target_is_directory=True)
        for split, ids in plan.items():
            (image_sets/f'{split}.txt').write_text('\n'.join(ids)+'\n', encoding='utf-8')
        print(f'{name}: train={len(plan["train"])} val={len(plan["val"])} | {target}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'data')
    a = p.parse_args()
    try:
        prepare(a.source, a.output)
    except (OSError, ValueError) as e:
        p.exit(1, f'{e}\n')
