import torch
import os
import numpy as np
import random


def seed_everything(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True


def return_device():
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	return device


def mixup_data(x, z, y, params):
	if params['mixup_alpha'] > 0:
		lam = np.random.beta(
			params['mixup_alpha'], params['mixup_alpha']
		)
	else:
		lam = 1

	batch_size = x.size()[0]
	if params['device'].type == 'cuda':
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)

	mixed_x = lam * x + (1 - lam) * x[index, :]
	mixed_z = lam * z + (1 - lam) * z[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, mixed_z, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def accuracy(output, labels):
	output = torch.softmax(output, 1)
	accuracy = (output.argmax(dim=1) == labels).float().mean()
	return accuracy
