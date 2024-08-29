PYTHONPATH=. python dinov2/run/train/train.py \
    --nodes 1 \
    --ngpus 1 \
    --config-file dinov2/configs/train/vitl16_short.yaml \
    --output-dir ./Outputs \
    train.dataset_path=ImageNet:split=TRAIN:root=dataset/tiny-imagenet-200:extra=dataset/tiny-imagenet-200/metadata
