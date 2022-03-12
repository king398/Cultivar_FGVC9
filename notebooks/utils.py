import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms

train_transforms = transforms.Compose([transforms.ToPILImage(),

                                transforms.ToTensor()])

def get_valid_transforms(DIM):
	return albumentations.Compose(
		[
			albumentations.Resize(DIM, DIM),

			ToTensorV2(p=1.0)
		]
	)

# create patches for a given image - adapted from https://www.kaggle.com/remekkinas/step-2-find-numbers-no-model-required
