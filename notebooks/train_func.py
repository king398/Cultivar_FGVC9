from tqdm.auto import tqdm
import torch
from torch.cuda.amp import autocast
from utils import *
import numpy as np
import gc
from augmentations import *


def train_fn(train_loader, model, criterion, optimizer, epoch, cfg, scheduler=None):
    device = torch.device(cfg['device'])
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    # if epoch < 5:
    # cfg['cutmix'] = False
    # else:
    # cfg['cutmix'] = True

    for i, (images, target) in enumerate(stream, start=1):
        if cfg['mixup']:
            images, target_a, target_b, lam = mixup_data(images, target, cfg['mixup_alpha'])
            images = images.to(device, non_blocking=True, dtype=torch.float)
            target_a = target_a.to(device)
            target_b = target_b.to(device)
        elif cfg['cutmix']:
            images, target_cut = cutmix(images, target, cfg['cutmix_alpha'])
            images = images.to(device, non_blocking=True, dtype=torch.float)
        else:
            images = images.to(device, non_blocking=True)
            target = target.to(device).long()

        with autocast():
            output = model(images).float()

        if cfg['mixup']:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        if cfg['cutmix']:
            loss = criterion(output, target_cut[0].cuda()) * target_cut[2] + criterion(output, target_cut[1].cuda()) * (
                        1. - target_cut[2].cuda())
        else:
            loss = criterion(output, target)

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
    accuracy_list = []
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)

            target = target.to(device, non_blocking=True).long()
            with autocast():
                output = model(images).float()
            loss = criterion(output, target)

            accuracy = accuracy_score(output, target)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")
            accuracy_list.append(accuracy)
    return np.mean(accuracy_list)


def inference_fn(test_loader, model, cfg):
    device = torch.device(cfg['device'])
    model.eval()
    stream = tqdm(test_loader)
    preds = None
    with torch.no_grad():
        for i, images in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)

            with autocast():
                output = model(images)

            pred = torch.softmax(output, 1).detach().cpu()
            if preds is None:
                preds = pred
            else:
                preds = torch.cat((preds, pred))
            del pred
            gc.collect()

    return preds


def oof_fn(test_loader, model, cfg):
    device = torch.device(cfg['device'])
    model.eval()
    stream = tqdm(test_loader)
    ids = []
    target = []
    preds = None
    probablitys = None
    accuracy_list = []
    with torch.no_grad():
        for i, (images, label, id) in enumerate(stream, start=1):
            ids.extend(id)
            target.extend(label)
            images = images.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).long()
            with autocast():
                output = model(images).float()
            probablity = torch.softmax(output, 1).detach().cpu()
            if probablitys is None:
                probablitys = probablity
            else:
                torch.cat((probablitys, probablity))
            pred = torch.argmax(output, 1).detach().cpu()
            if preds is None:
                preds = pred
            else:
                preds = torch.cat((preds, pred))
            accuracy_list.append(accuracy_score(output, label))
    return ids, target, preds.detach().cpu().numpy(), probablitys.detach().cpu().numpy(), np.mean(accuracy_list)


def clip_extract(loader, model, device):
    model.eval()
    stream = tqdm(loader)
    features = None
    with torch.no_grad():
        for i, images in enumerate(stream, start=1):
            images = images.to(device, non_blocking=True)

            with autocast():
                output = model.encode_image(images)

            if features is None:
                features = output
            else:
                features = torch.cat((features, output))
            del output
            gc.collect()
    return features.detach().cpu().numpy()


def nn_train(model, loader, device, criterion, optimizer, epoch, scheduler=None):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(loader)
    for i, (features, labels) in enumerate(stream, start=1):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast():
            output = model(features).float()
        loss = criterion(output, labels)
        accuracy = accuracy_score(output, labels)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")


def nn_valid(model, loader, device, criterion, epoch):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(loader)
    accuracy_list = []
    with torch.no_grad():
        for i, (features, labels) in enumerate(stream, start=1):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast():
                output = model(features).float()
            loss = criterion(output, labels)
            accuracy = accuracy_score(output, labels)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)

            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")
            accuracy_list.append(accuracy)
    return np.mean(accuracy_list)
