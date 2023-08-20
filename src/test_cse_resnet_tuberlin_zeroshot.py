import argparse
import os
import pickle
from TUBerlin import TUBerlinDataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import datetime
import random
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cdist
import pretrainedmodels
import torch.nn.functional as F
from ResnetModel import CSEResnetModel_KDHashing
from itq.quantizer import IterativeQuantizer

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch ResNet Model for TUBerlin mAP Testing')

# ======================== Architecture ===========================
parser.add_argument('--arch', '-a', metavar='ARCH', default='cse_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: se_resnet50)')

parser.add_argument('--num_classes', metavar='N', type=int, default=220,
                    help='number of classes (default: 220)')
parser.add_argument('--num_hashing', metavar='N', type=int, default=64,
                    help='number of hashing dimension (default: 64)')

# ======================== Testing ==============================
parser.add_argument('--batch_size', default=50, type=int, metavar='N',
                    help='number of samples per batch')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')

parser.add_argument('--resume_dir',
                    default='../cse_resnet50/checkpoint/',
                    type=str, metavar='PATH',
                    help='dir of model checkpoint (default: none)')
parser.add_argument('--resume_file',
                    default='model_best.pth.tar',
                    type=str, metavar='PATH',
                    help='file name of model checkpoint (default: none)')

parser.add_argument('--ems-loss', dest='ems_loss', action='store_true',
                    help='use ems loss for the training')
parser.add_argument('--precision', action='store_true', help='report precision@100')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model')

# ======================== DataSet ==============================
parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot', type=str,
                    help='zeroshot version for training and testing (default: zeroshot)')

# ======================== SAKE ===========================
parser.add_argument('--kd_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for kd loss (default: 1)')
parser.add_argument('--kdneg_lambda', metavar='LAMBDA', default='0.3', type=float,
                    help='lambda for semantic adjustment (default: 0.3)')
parser.add_argument('--sake_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for total SAKE loss (default: 1)')

parser.add_argument('--cls_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for cross entropy loss (default: 1)')

# ======================== Cross-batch Metric Learning ===========================
parser.add_argument('--K', metavar='K', default=128, type=int,
                    help='size of queue')
parser.add_argument('--cbm_start_iter', default=9625, type=int,
                    help='start iteration for cbm')
parser.add_argument('--cbm_lambda', metavar='LAMBDA', default=0.01, type=float,
                    help='lambda for Cross-batch Metric loss')

# ======================== Contrastive Learning ===========================
parser.add_argument('--temperature', metavar='LAMBDA', default=0.07, type=float,
                    help='lambda for temperature in contrastive learning')
parser.add_argument('--margin', metavar='MARGIN', default=0.2, type=float,
                    help='margin for contrastive loss')
parser.add_argument('--contrastive_lambda', metavar='LAMBDA', default=0.05, type=float,
                    help='lambda for contrastive loss')


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


SEED = 0
seed_torch(SEED)


def main():
    global args
    args = parser.parse_args()
    args.precision = True
    global savedir

    savedir = 'Tuberlin_' \
              + 'cls({})_'.format(args.cls_lambda) \
              + 'kd({}-{}-{})_'.format(args.kd_lambda, args.kdneg_lambda, args.sake_lambda) \
              + 'csm({}-{}-{})_'.format(args.K, args.xbm_start_iter, args.csm_lambda) \
              + 'cl({}-{}-{})_'.format(args.temperature, args.margin, args.contrastive_lambda) \
              + 'classes({})_'.format(args.num_classes) \
              + 'hashing({})'.format(args.num_hashing)

    savedir = os.path.join(args.resume_dir, savedir)

    if args.zero_version == 'zeroshot2':
        args.num_classes = 104

    feature_file = os.path.join(savedir, 'features_zero.pickle')
    if os.path.isfile(feature_file):
        print('load saved SBIR features')
        with open(feature_file, 'rb') as fh:
            predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
            predicted_features_query, binary_features_query, gt_labels_query, \
            scores, binary_scores = pickle.load(fh)
        if scores is None:
            scores = - cdist(predicted_features_query, predicted_features_gallery, metric='cosine')
            binary_scores = - cdist(binary_features_query, binary_features_gallery, metric='hamming')
    else:
        print('prepare SBIR features using saved model')
        predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
        predicted_features_query, binary_features_query, gt_labels_query, \
        scores, binary_scores = prepare_features()

    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mAP_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery)
        mAP_ls[gt_labels_query[fi]].append(mapi)

        mapi_binary = eval_AP_inner(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        mAP_ls_binary[gt_labels_query[fi]].append(mapi_binary)

    mAP = np.array([np.nanmean(maps) for maps in mAP_ls]).mean()
    mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls_binary]).mean()
    print('mAP - real value: {:.3f}, hash: {:.3f}'.format(mAP, mAP_binary))

    if args.precision:
        prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
        prec_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
        for fi in range(predicted_features_query.shape[0]):
            prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery)
            prec_ls[gt_labels_query[fi]].append(prec)

            prec_binary = eval_precision(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
            prec_ls_binary[gt_labels_query[fi]].append(prec_binary)

        prec = np.array([np.nanmean(pre) for pre in prec_ls]).mean()
        prec_binary = np.array([np.nanmean(pre) for pre in prec_ls_binary]).mean()
        print('Precision - real value: {:.3f}, hash: {:.3f}'.format(prec, prec_binary))


def prepare_features():
    # create model
    model = CSEResnetModel_KDHashing(args.arch, args.num_hashing, args.num_classes).cuda()
    print(str(datetime.datetime.now()) + ' model inited.')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # resume from a checkpoint
    if args.resume_file:
        resume = os.path.join(savedir, args.resume_file)
    else:
        resume = os.path.join(savedir, 'model_best.pth.tar')

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        args.start_epoch = checkpoint['epoch']

        save_dict = checkpoint['state_dict']
        model_dict = model.state_dict()

        trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
        print('trashed vars from resume dict:')
        print(trash_vars)

        resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
        # resume_dict['module.linear.cpars'] = save_dict['module.linear.weight']

        model_dict.update(resume_dict)
        model.load_state_dict(model_dict)

        # model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {} acc {:.4f})"
              .format(resume, checkpoint['epoch'], checkpoint['best_acc1'].item()))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        # return
    cudnn.benchmark = True

    # load data
    immean = [0.485, 0.456, 0.406]  # RGB channel mean for imagenet
    imstd = [0.229, 0.224, 0.225]

    transformations = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(immean, imstd)])

    tuberlin_zero_ext = TUBerlinDataset(split='zero',
                                        version='ImageResized_ready',
                                        zero_version=args.zero_version,
                                        transform=transformations, aug=False)
    zero_loader_ext = DataLoader(dataset=tuberlin_zero_ext, batch_size=args.batch_size,
                                 shuffle=False, num_workers=3)

    tuberlin_zero = TUBerlinDataset(split='zero',
                                    version='png_ready',
                                    zero_version=args.zero_version,
                                    transform=transformations, aug=False)
    zero_loader = DataLoader(dataset=tuberlin_zero, batch_size=args.batch_size,
                             shuffle=False, num_workers=3)
    print(str(datetime.datetime.now()) + ' data loaded.')

    predicted_features_gallery, gt_labels_gallery = get_features(zero_loader_ext, model)
    predicted_features_query, gt_labels_query = get_features(zero_loader, model, 0)
    scores = - cdist(predicted_features_query, predicted_features_gallery, metric='cosine')

    itq = IterativeQuantizer(num_bits=args.num_hashing, num_iterations=50)
    itq.fit(predicted_features_gallery)
    binary_features_query = itq.quantize(predicted_features_query)
    binary_features_gallery = itq.quantize(predicted_features_gallery)

    binary_scores = - cdist(binary_features_query, binary_features_gallery, metric='hamming')
    print('euclidean distance calculated')

    with open(os.path.join(savedir, 'features_zero.pickle'), 'wb') as fh:
        pickle.dump([predicted_features_gallery, binary_features_gallery, gt_labels_gallery,
                     predicted_features_query, binary_features_query, gt_labels_query,
                     scores, binary_scores], fh)

    return predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
           predicted_features_query, binary_features_query, gt_labels_query, \
           scores, binary_scores


def get_features(data_loader, model, tag=1):
    # switch to evaluate mode
    model.eval()
    features_all = []
    targets_all = []
    # avgpool = nn.AvgPool2d(7, stride=1).cuda()
    avgpool = nn.AdaptiveAvgPool2d(1).cuda()
    for i, (input, target, wv) in enumerate(data_loader):
        if i % 10 == 0:
            print(i, end=' ', flush=True)

        tag_input = (torch.ones(input.size()[0], 1) * tag).cuda()
        input = torch.autograd.Variable(input, requires_grad=False).cuda()

        # compute output
        # features = avgpool(model.module.features(input, tag_input)).cpu().detach().numpy()
        with torch.no_grad():
            features = model.original_model.features(input, tag_input)

            if args.pretrained:
                features = model.original_model.avg_pool(features)
                features = features.view(features.size(0), -1)
            else:
                features = model.original_model.hashing(features)

        features = F.normalize(features)
        features = features.cpu().detach().numpy()
        # features = features.reshape(input.size()[0],-1)
        # print(features.shape)
        # print(target.numpy().shape)
        # break
        features_all.append(features.reshape(input.size()[0], -1))
        targets_all.append(target.detach().numpy())
    print('')

    features_all = np.concatenate(features_all)
    targets_all = np.concatenate(targets_all)
    print('Features ready: {}, {}'.format(features_all.shape, targets_all.shape))

    return features_all, targets_all


def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]
    fp = np.logical_not(tp)

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap


def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def eval_precision(inst_id, scores, gt_labels, top=100):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]

    top = min(top, tot)

    sort_idx = np.argsort(-scores)
    return np.sum(pos_flag[sort_idx][:top]) / top


if __name__ == '__main__':
    main()
