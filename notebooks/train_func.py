from tqdm.auto import tqdm
import torch
from torch.cuda.amp import autocast
from utils import *


def train_fn(train_loader, model, criterion, optimizer, epoch, cfg, scheduler=None):
	device = torch.device(cfg['device'])
	metric_monitor = MetricMonitor()
	model.train()
	stream = tqdm(train_loader)
	for i, (images, target) in enumerate(stream, start=1):
		images = images.to(device, non_blocking=True)
		if cfg['task'] == "regression":
			target = target.to(device, non_blocking=True).float()
		else:
			target = target.to(device, non_blocking=True).long()

		with autocast():
			output = model(images).squeeze()
		loss = criterion(output, target)
		if cfg['task'] == "regression":
			accuracy = accuracy_score_regress(output, target)
		else:
			accuracy = accuracy_score(output, target)

		metric_monitor.update("Loss", loss.item())
		metric_monitor.update("Accuracy", accuracy)
		loss.backward()
		optimizer.step()
		if scheduler is not None:
			scheduler.step()

		optimizer.zero_grad()
		stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")


def validate_fn(val_loader, model, criterion, epoch, cfg):
	device = torch.device(cfg['device'])
	metric_monitor = MetricMonitor()
	model.eval()
	stream = tqdm(val_loader)
	with torch.no_grad():
		for i, (images, target) in enumerate(stream, start=1):
			images = images.to(device, non_blocking=True)
			if cfg['task'] == "regression":
				target = target.to(device, non_blocking=True).float()
			else:
				target = target.to(device, non_blocking=True).long()
			with autocast():
				output = model(images).squeeze()
			loss = criterion(output, target)
			if cfg['task'] == "regression":
				accuracy = accuracy_score_regress(output, target)
			else:
				accuracy = accuracy_score(output, target)
			metric_monitor.update("Loss", loss.item())
			metric_monitor.update("Accuracy", accuracy)
			stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")
