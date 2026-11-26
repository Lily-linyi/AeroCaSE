"""AeroCaSE checkpoint evaluation entry point."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='Evaluate an AeroCaSE checkpoint')
    parser.add_argument('checkpoint', help='Model checkpoint')
    parser.add_argument('--rein-root', type=Path, help='Path containing the external rein package')
    parser.add_argument('--backbone', type=Path, help='Frozen VFM weights for adaptation-only checkpoints')
    parser.add_argument('--config', type=Path, help='Override configuration embedded in the checkpoint')
    parser.add_argument('--check-only', action='store_true', help='Strict loading check without inference')
    parser.add_argument('--data-root', type=Path, help='Dataset root')
    parser.add_argument('--split', default='val')
    parser.add_argument('--img-dir', help='Image subdirectory relative to the dataset root')
    parser.add_argument('--ann-dir', help='Train-ID annotation subdirectory relative to the dataset root')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--work-dir', type=Path, default=Path('work_dirs/test'))
    parser.add_argument('--report', type=Path, help='Save loading status and metrics as JSON')
    parser.add_argument('--save-predictions', action='store_true')
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument('--resize', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    scale.add_argument('--native', action='store_true', help='Use native image dimensions')
    args = parser.parse_args()
    if args.rein_root:
        sys.path.insert(0, str(args.rein_root.resolve()))
    import torch
    torch.set_num_threads(2)
    if not args.check_only:
        if args.data_root is None:
            parser.error('--data-root is required for evaluation')
        if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
            parser.error('Evaluation requires an available CUDA GPU')
    from core.checkpoint import load_model
    model, config, report = load_model(args.checkpoint, args.backbone, args.config)
    if not args.check_only:
        from core.evaluation import evaluate
        report.update(evaluate(model, config, data_root=args.data_root,
                               device=args.device, split=args.split, resize=args.resize,
                               native=args.native, work_dir=args.work_dir,
                               save_predictions=args.save_predictions,
                               img_dir=args.img_dir, ann_dir=args.ann_dir))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    destination = args.report or (None if args.check_only else args.work_dir / 'metrics.json')
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
