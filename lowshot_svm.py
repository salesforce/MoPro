from __future__ import print_function

import os
import sys
import time
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, datasets

from resnet import *

from DataLoader.Places_dataset import Places205
from DataLoader.VOC_dataset import Voc2007Classification

from sklearn.svm import LinearSVC
import numpy as np
import random


parser = argparse.ArgumentParser('argument for training')

parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--num_workers', type=int, default=20, help='num of workers to use')

# model definition
parser.add_argument('--model_path', type=str, default='', help='the model to test')
parser.add_argument('--supervise', default=False, action='store_true', help='whether to use supervised pretrained model')
parser.add_argument('--cost', type=float, default=0.5, help='cost parameter for SVM')

# dataset
parser.add_argument('--dataset', type=str, default='voc', choices=['places', 'voc'])
parser.add_argument('--voc-path', type=str, default='')
parser.add_argument('--places-path', type=str, default='')

# seed
parser.add_argument('--seed', default=0, type=int)    



def main():
   
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
        
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)        
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),        
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset=='voc':
        train_dataset = Voc2007Classification(args.voc_path,set='trainval',transform = transform)
        val_dataset = Voc2007Classification(args.voc_path,set='test',transform = transform)
        
    elif args.dataset=='places':
        train_dataset = Places205(args.places_path, 'train', transform = transform)        
        val_dataset = Places205(args.places_path, 'val', transform = transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    model = resnet50(encoder=True)       
    
    if not args.supervise:
        print('==> loading pre-trained model')
        ckpt = torch.load(args.model_path)
        state_dict = ckpt['state_dict']

        # rename pre-trained keys
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') and not k.startswith('module.encoder_q.classifier'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]  
            
        model.load_state_dict(state_dict,strict=False)
        print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
    else:
        print("==> use supervised pre-trained model")

    model = model.cuda()
    model.eval()   
        
    test_feats = []
    test_labels = []    
    print('==> calculate test features')
    for idx, (inputs, target) in enumerate(val_loader):
        inputs = inputs.cuda()
        feat = model(inputs)
        feat = feat.detach().cpu()
        test_feats.append(feat)
        test_labels.append(target)
        
    test_feats = torch.cat(test_feats,0).numpy()
    test_labels = torch.cat(test_labels,0).numpy()
               
    test_feats_norm = np.linalg.norm(test_feats, axis=1)
    test_feats = test_feats / (test_feats_norm + 1e-5)[:, np.newaxis]
    
    result={}

    for k in [1,2,4,8,16]: #number of samples per-class

        avg_map = []
        for run in range(5): # 5 runs
            print('==> re-sampling training data')
            train_dataset.convert_low_shot(k)    
            print(len(train_dataset))

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            train_feats = []
            train_labels = []
            print('==> calculate train features')
            for idx, (inputs, target) in enumerate(train_loader):
                inputs = inputs.cuda()
                feat = model(inputs)
                feat = feat.detach()  

                train_feats.append(feat)                
                train_labels.append(target)

            train_feats = torch.cat(train_feats,0).cpu().numpy()
            train_labels = torch.cat(train_labels,0).cpu().numpy()

            train_feats_norm = np.linalg.norm(train_feats, axis=1)
            train_feats = train_feats / (train_feats_norm + 1e-5)[:, np.newaxis]

            print('==> training SVM Classifier')
            if args.dataset=='places':        
                clf = LinearSVC(random_state=0, tol=1e-4, C=args.cost, dual=True, max_iter=2000)
                clf.fit(train_feats, train_labels)

                prediction = clf.predict(test_feats)
                print('==> testing SVM Classifier')
                accuracy = 100.0*(prediction==test_labels).sum()/len(test_labels)
                avg_map.append(accuracy)
                print('==> Run%d accuracy is %.2f: '%(run,accuracy))

            elif args.dataset=='voc':
                cls_ap = np.zeros((20, 1))
                test_labels[test_labels==0] = -1 
                train_labels[train_labels==0] = -1 
                for cls in range(20):
                    clf = LinearSVC(
                        C=args.cost, class_weight={1: 2, -1: 1}, intercept_scaling=1.0,
                        penalty='l2', loss='squared_hinge', tol=1e-4,
                        dual=True, max_iter=2000,random_state=0)
                    clf.fit(train_feats, train_labels[:,cls])

                    prediction = clf.decision_function(test_feats)                                      
                    P, R, score, ap = get_precision_recall(test_labels[:,cls], prediction)
                    cls_ap[cls][0] = ap*100
                mean_ap = np.mean(cls_ap, axis=0)

                print('==> Run%d mAP is %.2f: '%(run,mean_ap))
                avg_map.append(mean_ap)

        avg_map = np.asarray(avg_map)     
        print('Average ap is: %.2f' %(avg_map.mean()))
        print('Std is: %.2f' %(avg_map.std()))

        result[k] = avg_map.mean()
    print(result)  
    
    
def calculate_ap(rec, prec):
    """
    Computes the AP under the precision recall curve.
    """
    rec, prec = rec.reshape(rec.size, 1), prec.reshape(prec.size, 1)
    z, o = np.zeros((1, 1)), np.ones((1, 1))
    mrec, mpre = np.vstack((z, rec, o)), np.vstack((z, prec, z))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = 0
    for i in indices:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def get_precision_recall(targets, preds):
    """
    [P, R, score, ap] = get_precision_recall(targets, preds)
    Input    :
        targets  : number of occurrences of this class in the ith image
        preds    : score for this image
    Output   :
        P, R   : precision and recall
        score  : score which corresponds to the particular precision and recall
        ap     : average precision
    """
    # binarize targets
    targets = np.array(targets > 0, dtype=np.float32)
    tog = np.hstack((
        targets[:, np.newaxis].astype(np.float64),
        preds[:, np.newaxis].astype(np.float64)
    ))
    ind = np.argsort(preds)
    ind = ind[::-1]
    score = np.array([tog[i, 1] for i in ind])
    sortcounts = np.array([tog[i, 0] for i in ind])

    tp = sortcounts
    fp = sortcounts.copy()
    for i in range(sortcounts.shape[0]):
        if sortcounts[i] >= 1:
            fp[i] = 0.
        elif sortcounts[i] < 1:
            fp[i] = 1.
    P = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
    numinst = np.sum(targets)
    R = np.cumsum(tp) / numinst
    ap = calculate_ap(R, P)
    return P, R, score, ap   


if __name__ == '__main__':
    main()
