import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, CosineAnnealingLR
from collections import defaultdict


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


def accuracy_score(output, labels):
	output = torch.softmax(output, 1)
	accuracy = (output.argmax(dim=1) == labels).float().mean()
	accuracy = accuracy.detach().cpu().numpy()
	return accuracy


class MetricMonitor:
	def __init__(self, float_precision=3):
		self.float_precision = float_precision
		self.reset()

	def reset(self):
		self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

	def update(self, metric_name, val):
		metric = self.metrics[metric_name]

		metric["val"] += val
		metric["count"] += 1
		metric["avg"] = metric["val"] / metric["count"]

	def __str__(self):
		return " | ".join(
			[
				"{metric_name}: {avg:.{float_precision}f}".format(
					metric_name=metric_name, avg=metric["avg"],
					float_precision=self.float_precision
				)
				for (metric_name, metric) in self.metrics.items()
			]
		)


def get_scheduler(optimizer, scheduler_params):
	if scheduler_params['scheduler_name'] == 'CosineAnnealingWarmRestarts':
		scheduler = CosineAnnealingWarmRestarts(
			optimizer,
			T_0=scheduler_params['T_0'],
			eta_min=scheduler_params['min_lr'],
			last_epoch=-1
		)


	elif scheduler_params['scheduler_name'] == 'CosineAnnealingLR':
		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=scheduler_params['T_max'],
			eta_min=scheduler_params['min_lr'],
			last_epoch=-1
		)
	return scheduler
