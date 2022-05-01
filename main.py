import argparse
import sys
import torch
from torch import cuda
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torchtoolbox.tools import mixup_data, mixup_criterion
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy
import torchmetrics
#  ours
from dataset import cultivar, loader_fold
from trip_loss import *
from begin import *

parser = argparse.ArgumentParser()
parser.add_argument('--note', default='xx', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--aug_mode', default='T', type=str)

parser.add_argument('--gpu_parallel', default=False, type=bool)
parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--model_name', default='tresnet_m', type=str)
parser.add_argument('--num_epoch', default=40, type=int)
parser.add_argument('--img_size', default=512, type=int)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--num_worker', default=8, type=int)
parser.add_argument('--k_fold', default=5, type=int)
parser.add_argument('--bs', default=32, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
args = parser.parse_args()


class our_model(nn.Module):
    def __init__(self, model_name, num_classes, dim1=1024, pretrained=True):
        super(our_model, self).__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.in_features = self.model.get_classifier().in_features

        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.pooling = GeM()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.in_features, dim1)
        self.bn1 = nn.BatchNorm1d(dim1)
        self.classifier = nn.Linear(dim1, self.num_classes)

    def extract(self, image):
        bs = image.size()[0]
        x = self.model(image)
        x = self.pooling(x).view(bs, -1)

        x = self.relu(x)
        x = self.fc1(x)
        feature = self.bn1(x)
        return feature

    def forward(self, image, ret_f=False):
        feature = self.extract(image)
        logits = self.classifier(feature)
        if ret_f:
            return logits, feature
        else:
            return logits


class our_model1(nn.Module):
    def __init__(self, cfg, dim1=1280, pretrained=True):
        super(our_model1, self).__init__()
        self.num_classes = cfg['target_size']
        self.model = timm.create_model(cfg['model'], pretrained=pretrained)
        self.in_features = self.model.get_classifier().in_features

        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.pooling = GeM()
        self.fc1 = nn.Linear(self.in_features, self.in_features)
        self.bn1 = nn.BatchNorm1d(self.in_features)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(self.in_features, dim1)
        self.bn2 = nn.BatchNorm1d(dim1)
        self.classifier = nn.Linear(dim1, self.num_classes)

    def extract(self, image):
        bs = image.size()[0]
        x = self.model(image)
        x = self.pooling(x).view(bs, -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        feature = self.bn2(x)
        return feature

    def forward(self, image, ret_f=False):
        feature = self.extract(image)
        feature = self.relu(feature)
        logits = self.classifier(feature)
        if ret_f:
            return logits, feature
        else:
            return logits


def train_phase(model, train_loader, optimizer, epoch, args, fold_i, num_per_cls, metric):
    train_loss = 0.0
    scaler = cuda.amp.GradScaler()

    loss_function = LabelSmoothingCrossEntropy(smoothing=0.1)
    # loss_function = nn.CrossEntropyLoss().cuda()
    metric_loss = TripletLoss()

    for index, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()

        data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
        with cuda.amp.autocast():
            output, feature = model(data, ret_f=True)
            loss_ce = loss_function(output, label)
            loss_metric = metric_loss(feature, label)

            loss = loss_ce + loss_metric

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metric(output.argmax(1), label)
        train_loss += loss.item()

        sys.stdout.write('\r')
        sys.stdout.write('Epoch:[%2d/%2d]\t   iter:[%4d/%4d]\t   loss_cls/[Trip/Arc]: %.3f/%.3f/%.3f\t   lr: %.8f' %
                         (epoch + 1, args.num_epoch, index + 1, len(train_loader), loss_ce.item(), loss_metric.item(),
                          0, optimizer.param_groups[0]['lr']))
        sys.stdout.flush()

        schedule_lr.step()

    sys.stdout.write('\n')
    train_loss = train_loss / len(train_loader)
    pprint('Fold: [%2d/%2d]\t | Epoch: [%2d/%2d]\t  |  avg. train_acc:%.4f\t  train_loss:%.4f\t lr: %.10f' % (
        fold_i + 1, args.k_fold, epoch + 1, args.num_epoch, metric.compute(), train_loss,
        optimizer.param_groups[0]['lr']), test_log)

    metric.reset()


def val_phase(model, val_loader, epoch, args, fold_i, metric):
    with torch.no_grad():
        for data_v, label_v in val_loader:
            data_v, label_v = data_v.to(device, non_blocking=True), label_v.to(device, non_blocking=True)
            logits_v = model(data_v)

            metric(logits_v.argmax(1), label_v)
    val_acc = metric.compute()

    pprint('Fold: [%2d/%2d]\t | Epoch: [%2d/%2d]\t  |  val_acc:%.4f\t' %
           (fold_i + 1, args.k_fold, epoch + 1, args.num_epoch, val_acc), test_log)
    metric.reset()
    return val_acc


if __name__ == '__main__':
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if len(args.gpu) == 1:
        device_ = [0]
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.gpu_parallel = False
    else:
        num_gpu = args.gpu.split(',')
        device_ = [i for i in range(len(num_gpu))]
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        args.gpu_parallel = True

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # ===============  set dir_path ================
    test_path = get_logname(args)
    test_log = open(test_path + '.txt', 'w')
    model_pt = test_path

    # ================ train and val =================
    skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    best_acc = 0.0

    df_all, df_t2t = loader_fold(mode='train', t2t=True)
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(df_all['image'], df_all['cultivar_index'])):
        # initialize dataloader
        train, valid = df_all.iloc[train_idx], df_all.iloc[val_idx]
        train_image_path, train_labels = train['file_path'].values.tolist(), train['cultivar_index'].values.tolist()
        val_image_path, val_labels = valid['file_path'].values.tolist(), valid['cultivar_index'].values.tolist()
        # test_to_train
        # t2t_image_path, t2t_labels = df_t2t['filename'].values.tolist(), df_t2t['cultivar_index'].values.tolist()
        #
        # for i in range(len(t2t_labels)):
        #     train_image_path.append(t2t_image_path[i])
        #     train_labels.append(int(t2t_labels[i]))

        # print(type(train_image_path))
        # for i in range(len(val_image_path)):
        #     train_image_path.append(val_image_path[i])
        #     train_labels.append(val_labels[i])
        num_per_cls = []
        for i in range(100):
            num_per_cls.append(train_labels.count(i))

        num_per_cls = torch.tensor(num_per_cls).cuda()
        print(num_per_cls)

        # =============================== load dataloader ============================
        train_dataset = cultivar(args, mode='train', aug_mode=args.aug_mode, image_paths=train_image_path,
                                 labels=train_labels)
        data_source = [tuple([train_image_path[i], train_labels[i], i]) for i in range(len(train_labels))]
        train_sampler = RandomIdentitySampler(data_source, num_instances=4, batch_size=args.bs)
        train_loader = DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler, num_workers=args.num_worker,
                                  pin_memory=True)

        val_dataset = cultivar(args, mode='val', aug_mode=args.aug_mode, image_paths=val_image_path, labels=val_labels)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_worker,
                                pin_memory=True)

        # =============================== load our_model ============================
        model = our_model1(args.model_name, num_classes=args.num_classes, pretrained=True)
        pprint(str(model), test_log)
        # model_dict = torch.load('checkpoints/' + args.model_name + '/2022-04-26 21:39:58_896_40_32_0.0001.pt')
        # model.load_state_dict(model_dict)
        # print('load checkpoint!....')

        if args.gpu_parallel:
            model = torch.nn.DataParallel(model, device_ids=device_)
        model.to(device)

        args.steps_per_epoch = len(train_loader)
        args.max_lr = 3e-4
        args.pct_start = 0.15
        args.div_factor = 1.0e+3
        args.final_div_factor = 1.0e+3

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # sch_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch + 1, eta_min=1e-7)
        schedule_lr = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=args.num_epoch,
                                                          steps_per_epoch=args.steps_per_epoch,
                                                          max_lr=args.max_lr, pct_start=args.pct_start,
                                                          div_factor=args.div_factor,
                                                          final_div_factor=args.final_div_factor)
        metric = torchmetrics.Accuracy(threshold=0.5, num_classes=args.num_classes).to(device)

        pprint('--load_path-- | {}'.format(test_path), test_log)
        pprint(str(args), test_log)
        pprint('Fold: [%2d/%2d]\t | train num: %d\t | val num: %d\t \n' % (
            fold_i + 1, args.k_fold, len(train_image_path), len(val_image_path)), test_log)
        # ============================ Train ===============================
        for epoch in range(args.num_epoch):
            model.train()
            train_phase(model, train_loader, optimizer, epoch, args, fold_i, num_per_cls, metric)

            model.eval()
            val_acc = val_phase(model, val_loader, epoch, args, fold_i, metric)
            # ======== save model =======
            if val_acc > best_acc:
                best_acc = val_acc
                if args.gpu_parallel:
                    torch.save(model.module.state_dict(), model_pt + '.pt')
                else:
                    torch.save(model.state_dict(), model_pt + '.pt')
                pprint('======= Save model! =======\n', test_log)
            else:
                pprint('===========================\n', test_log)

        break
    pprint('best_acc: {}'.format(best_acc), test_log)
    print(str(args))
