import albumentations
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(DIM):
	return albumentations.Compose(
		[
			albumentations.Resize(DIM, DIM),
			albumentations.Normalize(
				mean=[0.3511794, 0.37462908, 0.2873578],
				std=[0.20823358, 0.2117826, 0.16226698],
			),
			albumentations.HorizontalFlip(),
			albumentations.VerticalFlip(),
			ToTensorV2(p=1.0)
		]
	)


def get_valid_transforms(DIM):
	return albumentations.Compose(
		[
			albumentations.Resize(DIM, DIM),
			albumentations.Normalize(
				mean=[0.3511794, 0.37462908, 0.2873578],
				std=[0.20823358, 0.2117826, 0.16226698],
			),
			ToTensorV2(p=1.0)
		]
	)


def get_test_transforms(DIM):
	return albumentations.Compose(
		[
			albumentations.Resize(DIM, DIM),
			albumentations.Normalize(
				mean=[0.3511794, 0.37462908, 0.2873578],
				std=[0.20823358, 0.2117826, 0.16226698],
			),
			ToTensorV2(p=1.0)
		]
	)
