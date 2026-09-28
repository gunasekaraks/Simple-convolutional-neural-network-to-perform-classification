"""Download TF Flowers and save reproducible, stratified 64x64 splits."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw


def prepare(output=Path('data'), seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset, info = tfds.load(
        'tf_flowers:3.0.1', split='train', as_supervised=True,
        with_info=True, shuffle_files=False, data_dir=str(output / 'tfds'),
        try_gcs=False,
    )
    print(info)
    names = info.features['label'].names
    images, labels = [], []
    for image, label in dataset:
        # Bilinear antialiasing reduces artifacts during downsampling.
        resized = tf.image.resize(image, [64, 64], method='bilinear', antialias=True)
        images.append(np.clip(np.rint(resized.numpy()), 0, 255).astype(np.uint8))
        labels.append(int(label.numpy()))
    images, labels = np.stack(images), np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = {name: [] for name in ('train', 'validation', 'test')}
    for label in range(len(names)):
        ids = rng.permutation(np.flatnonzero(labels == label))
        n_train, n_val = round(len(ids) * .70), round(len(ids) * .15)
        for split, part in zip(indices, (ids[:n_train], ids[n_train:n_train+n_val],
                                         ids[n_train+n_val:])):
            indices[split].extend(part.tolist())
    for split in indices:
        indices[split] = rng.permutation(indices[split])
    combined = np.concatenate(list(indices.values()))
    assert len(combined) == len(labels) == info.splits['train'].num_examples
    assert len(np.unique(combined)) == len(labels), 'Split overlap or missing samples'
    summary = {
        'dataset': 'tf_flowers', 'version': str(info.version), 'seed': seed,
        'class_names': names, 'image_shape': [64, 64, 3],
        'storage_dtype': 'uint8', 'storage_range': [0, 255],
        'resize': 'bilinear with antialiasing; direct resize (aspect ratio may change)',
        'split_method': 'seeded per-class shuffle; round 70% and 15%; remainder test',
        'total_images': len(labels), 'splits': {},
    }
    with (output / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_index', 'split', 'label', 'class_name'])
        for split, ids in indices.items():
            assert images[ids].shape == (len(ids), 64, 64, 3)
            np.savez_compressed(output / f'{split}.npz', images=images[ids],
                                labels=labels[ids], source_indices=ids)
            counts = {name: int(np.sum(labels[ids] == i)) for i, name in enumerate(names)}
            summary['splits'][split] = {'count': len(ids), 'class_counts': counts}
            writer.writerows((int(i), split, int(labels[i]), names[labels[i]]) for i in ids)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # A small labeled preview uses training images only.
    preview = Image.new('RGB', (5 * 128, len(names) * 92), 'white')
    draw = ImageDraw.Draw(preview)
    for label, name in enumerate(names):
        ids = [i for i in indices['train'] if labels[i] == label][:5]
        for col, i in enumerate(ids):
            preview.paste(Image.fromarray(images[i]), (col * 128, label * 92))
            draw.text((col * 128, label * 92 + 66), name, fill='black')
    preview.save(output / 'training_preview.png')
    print(json.dumps(summary, indent=2))
    return summary


def load_datasets(data_dir='data', batch_size=32, normalize=True, seed=42):
    """Return train/validation/test datasets; only training is shuffled.

    normalize=True yields float32 [0,1] for custom CNNs. For pretrained
    models use normalize=False (float32 [0,255]) and that model's specific
    preprocessing function. Apply augmentation to training only, if desired.
    """
    result = []
    for split in ('train', 'validation', 'test'):
        with np.load(Path(data_dir) / f'{split}.npz') as saved:
            images = saved['images'].astype(np.float32)
            labels = saved['labels']
        if normalize:
            images /= 255.0
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        if split == 'train':
            dataset = dataset.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
        result.append(dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE))
    return tuple(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.output, args.seed)
