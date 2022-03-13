import albumentations
from albumentations.pytorch.transforms import ToTensorV2


def get_valid_transforms(DIM):
	return albumentations.Compose(
		[
			albumentations.Resize(DIM, DIM),
			albumentations.Normalize(
				mean=0.1307,
				std=0.3081,
			),
			ToTensorV2(p=1.0)
		]
	)


def get_train_transforms(DIM):
	return albumentations.Compose(
		[
			albumentations.Resize(DIM, DIM),
			albumentations.Normalize(
				mean=0.1307,
				std=0.3081,
			),
			ToTensorV2(p=1.0)
		]
	)
