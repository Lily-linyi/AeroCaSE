"""Evaluation of segmentation predictions at the ground-truth resolution."""
from copy import deepcopy
from pathlib import Path
import numpy as np


def update_confusion(histogram, target, prediction, ignore_index=255):
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if target.shape != prediction.shape:
        raise ValueError('Prediction and target must have identical native dimensions')
    n = histogram.shape[0]
    valid = (target != ignore_index) & (target >= 0) & (target < n)
    if np.any((prediction[valid] < 0) | (prediction[valid] >= n)):
        raise ValueError('Prediction contains invalid class IDs')
    histogram += np.bincount(n * target[valid] + prediction[valid],
                            minlength=n * n).reshape(n, n)


def summarize_confusion(histogram, classes):
    hit = np.diag(histogram).astype(float)
    actual = histogram.sum(axis=1)
    union = actual + histogram.sum(axis=0) - hit
    iou = np.divide(hit, union, out=np.full_like(hit, np.nan), where=union > 0)
    acc = np.divide(hit, actual, out=np.full_like(hit, np.nan), where=actual > 0)
    def percent(value):
        return None if not np.isfinite(value) else float(value * 100.)
    result = dict(mIoU=percent(np.nanmean(iou)), mAcc=percent(np.nanmean(acc)),
                  aAcc=percent(hit.sum() / actual.sum()) if actual.sum() else None,
                  per_class={name: dict(IoU=percent(iou[i]), Acc=percent(acc[i]))
                             for i, name in enumerate(classes)})
    small = [i for i, name in enumerate(classes) if name.lower() in ('person', 'vehicle')]
    if small:
        result['S_mIoU'] = percent(np.nanmean(iou[small]))
    return result


def build_dataset(config, data_root, split='val', resize=None, native=False,
                  metadata=None, img_dir=None, ann_dir=None):
    from mmseg.datasets import BaseSegDataset
    loader = config.get('test_dataloader') or config.get('val_dataloader')
    if loader is None:
        raise ValueError('Configuration must supply a test or validation pipeline')
    dataset_options = loader['dataset']
    while 'dataset' in dataset_options:
        dataset_options = dataset_options['dataset']
    pipeline = deepcopy(dataset_options['pipeline'])
    resize_types = {'Resize', 'ResizeShortestEdge', 'RandomChoiceResize'}
    if native or resize is not None:
        pipeline = [p for p in pipeline if p['type'].split('.')[-1] not in resize_types]
    if resize is not None:
        if min(resize) <= 0:
            raise ValueError('Resize dimensions must be positive')
        index = next(i for i, p in enumerate(pipeline)
                     if p['type'].split('.')[-1] == 'LoadImageFromFile')
        pipeline.insert(index + 1, dict(type='Resize', scale=tuple(resize), keep_ratio=True))
    # Loading GT after geometric resizing keeps annotations in native coordinates.
    annotations = [p for p in pipeline if p['type'].split('.')[-1] == 'LoadAnnotations']
    pipeline = [p for p in pipeline if p['type'].split('.')[-1] != 'LoadAnnotations']
    pack = next(i for i, p in enumerate(pipeline)
                if p['type'].split('.')[-1] == 'PackSegInputs')
    pipeline[pack:pack] = annotations or [dict(type='LoadAnnotations')]
    image_dir = Path(data_root) / (img_dir or str(Path(split) / 'img'))
    label_dir = Path(data_root) / (ann_dir or str(Path(split) / 'seg_trainid'))
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError('Image and label directories must exist; use --img-dir and --ann-dir')
    suffixes = {p.suffix.lower() for p in image_dir.rglob('*')
                if p.suffix.lower() in ('.png', '.jpg', '.jpeg')}
    if len(suffixes) != 1:
        raise ValueError('Image directory must contain one supported image extension')
    options = dict(data_root=str(data_root),
                   data_prefix=dict(img_path=str(image_dir.resolve()),
                                    seg_map_path=str(label_dir.resolve())),
                   pipeline=pipeline, test_mode=True,
                   img_suffix=next(iter(suffixes)),
                   seg_map_suffix=dataset_options.get('seg_map_suffix', '.png'),
                   reduce_zero_label=dataset_options.get('reduce_zero_label', False))
    if metadata and metadata.get('classes'):
        options['metainfo'] = metadata
    return BaseSegDataset(**options)


def evaluate(model, config, *, data_root, device, split='val', resize=None,
             native=False, work_dir='work_dirs/test', save_predictions=False,
             img_dir=None, ann_dir=None):
    import torch
    from PIL import Image
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('GPU evaluation requires an available CUDA device')
    dataset = build_dataset(config, data_root, split, resize, native,
                            model.dataset_meta, img_dir, ann_dir)
    classes = dataset.metainfo.get('classes')
    if not classes:
        classes = [str(i) for i in range(model.decode_head.num_classes)]
    if len(classes) != model.decode_head.num_classes:
        raise ValueError('Dataset and model class counts differ')
    histogram = np.zeros((len(classes), len(classes)), dtype=np.int64)
    destination = Path(work_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model = model.float().to(device).eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            batch = dict(inputs=[sample['inputs']], data_samples=[sample['data_samples']])
            output = model.test_step(batch)[0]
            prediction = output.pred_sem_seg.data.squeeze(0).cpu().numpy()
            target = sample['data_samples'].gt_sem_seg.data.squeeze(0).cpu().numpy()
            update_confusion(histogram, target, prediction)
            if save_predictions:
                relative = Path(dataset.get_data_info(index)['img_path']).relative_to(
                    (Path(data_root) / (img_dir or str(Path(split) / 'img'))).resolve())
                path = destination / 'predictions' / relative.with_suffix('.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(prediction.astype(np.uint8)).save(path)
            print(f'Evaluated {index + 1}/{len(dataset)}', flush=True)
    return dict(inference_executed=True, images=len(dataset),
                metrics=summarize_confusion(histogram, classes))
