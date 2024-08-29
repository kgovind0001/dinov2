from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="dataset/tiny-imagenet-200/", extra="dataset/tiny-imagenet-200/metadata")
    dataset.dump_extra()