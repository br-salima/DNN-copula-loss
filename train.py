import torch
import os
import numpy as np
from scipy import stats
import yaml
from argparse import ArgumentParser
import random
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import sys
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from network import NSSADNN
from IQADataset import IQADataset

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)





def get_indexNum(config, index, status):
    test_ratio = config['test_ratio']
    train_ratio = config['train_ratio']
    trainindex = index[:int(train_ratio * len(index))]
    testindex = index[int((1 - test_ratio) * len(index)):]
    train_index, val_index, test_index = [], [], []

    ref_ids = []
    for line0 in open("./data_copule/ref_ids.txt", "r"):
        line0 = float(line0[:-1])
        ref_ids.append(line0)
    ref_ids = np.array(ref_ids)

    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)
    if status == 'train':
        index = train_index
    if status == 'test':
        index = test_index
    if status == 'val':
        index = val_index

    return len(index)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="LIVE_Phase1")
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    args = parser.parse_args()
    ensure_dir('results')
    save_model = "./results/model.pth"  


    seed = random.randint(10000000, 99999999)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("seed:", seed)

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    index = []
    if args.dataset == "LIVE_Phase1":
        print("dataset: LIVE_Phase1")
        index = list(range(1, 21))
        random.shuffle(index)
    elif args.dataset == "TID2013":
        print("dataset: TID2013")
        index = list(range(0, 26))

    print('rando index', index)

    dataset = args.dataset
    valnum = get_indexNum(config, index, "val")
    testnum = get_indexNum(config, index, "test")

    train_dataset = IQADataset(dataset, config, index, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)
    print("shape of train_loader ")
    print(len(train_loader))
    val_dataset = IQADataset(dataset, config, index, "val")
    val_loader = torch.utils.data.DataLoader(val_dataset)

    test_dataset = IQADataset(dataset, config, index, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset)

    writer = SummaryWriter('runs/TensorBoard')
    model = NSSADNN().to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    torch.optim.lr_scheduler.StepLR(optimizer, 750, gamma=0.1, last_epoch=-1)

    best_SROCC = -1

    for epoch in range(args.epochs):
        # train
        model.train()
        LOSS_all = 0
        LOSS_NSS = 0
        LOSS_q = 0
        for i, (patches, (label, features)) in enumerate(train_loader):
            patches = patches.to(device)
            label = label.to(device)
            features = features.to(device).float()

            optimizer.zero_grad()
            outputs_q = model(patches)[0]
            outputs_NSS = model(patches)[1]

            loss_NSS = criterion(outputs_NSS, features)
            #print('epoch',epoch)
            #print('iteration',i)
            #print('loss_NSS',loss_NSS)
            loss_q = criterion(outputs_q, label)
            #print('loss_q',loss_q)
            loss = 23*loss_NSS + loss_q
            #print ('loss',loss)
            loss.backward()
            optimizer.step()
            LOSS_all = LOSS_all + loss.item()
            LOSS_NSS = LOSS_NSS + loss_NSS.item()
            LOSS_q = LOSS_q + loss_q.item()
          
        train_loss_all = LOSS_all / (i + 1)
        print ('train_loss_all',train_loss_all)
        train_loss_NSS = LOSS_NSS / (i + 1)
        print('train_loss_copula',train_loss_NSS)
        train_loss_q = LOSS_q / (i + 1)
        print('quality score loss',train_loss_q)
        writer.add_scalar('quality score loss', train_loss_q,  epoch * len(train_loader) + i)
        writer.add_scalar('copula features prediction loss',train_loss_NSS,  epoch * len(train_loader) + i)
        writer.add_scalar('total_loss',train_loss_NSS, epoch * len(train_loader) + i)
        writer.close()
        # val
        y_pred = np.zeros(valnum)
        y_val = np.zeros(valnum)
        model.eval()
        L = 0
        with torch.no_grad():   # Update the weights using gradient descent. Each parameter is a Tensor,
            for i, (patches, (label, features)) in enumerate(val_loader):
                y_val[i] = label.item()
                patches = patches.to(device)
                label = label.to(device)
                outputs_q = model(patches)[0]
                score = outputs_q.mean()
                y_pred[i] = score
                loss = criterion(score, label[0])
                L = L + loss.item()
        val_loss = L / (i + 1)

        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]
        val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]
        val_RMSE = np.sqrt(((y_pred - y_val) ** 2).mean())

        writer.add_scalar("validation/val_SROCC", val_SROCC,  epoch * len(val_loader) + i)
        writer.add_scalar("validation/val_PLCC", val_PLCC,epoch * len(val_loader) + i)
        writer.add_scalar("validation/val_KROCC", val_KROCC, epoch * len(val_loader) + i)
        writer.add_scalar("validation/val_RMSE", val_RMSE,epoch * len(val_loader) + i)

        # test
        y_pred = np.zeros(testnum)
        y_test = np.zeros(testnum)
        L = 0
        with torch.no_grad():
            for i, (patches, (label, features)) in enumerate(test_loader):
                y_test[i] = label.item()
                patches = patches.to(device)
                label = label.to(device)
                outputs_q = model(patches)[0]

                score = outputs_q.mean()
                y_pred[i] = score
                loss = criterion(score, label[0])
                L = L + loss.item()
        test_loss = L / (i + 1)
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
        writer.add_scalar("test/test_SROCC", SROCC,  epoch * len(val_loader) + i)
        writer.add_scalar("test/test_PLCC", PLCC,epoch * len(val_loader) + i)
        writer.add_scalar("test/test_KROCC", KROCC, epoch * len(val_loader) + i)
        writer.add_scalar("test/test_RMSE", RMSE,epoch * len(val_loader) + i)

        print("Epoch {} Valid Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(epoch,
                                                                                                             val_loss,
                                                                                                             val_SROCC,
                                                                                                             val_PLCC,
                                                                                                             val_KROCC,
                                                                                                             val_RMSE))
        print("Epoch {} Test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(epoch,
                                                                                                            test_loss,
                                                                                                            SROCC,
                                                                                                            PLCC,
                                                                                                            KROCC,
                                                                                                            RMSE))

        if val_SROCC > best_SROCC and epoch > 100:
            print("Update Epoch {} best valid SROCC".format(epoch))
            print("Valid Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(val_loss,
                                                                                                        val_SROCC,
                                                                                                        val_PLCC,
                                                                                                        val_KROCC,
                                                                                                        val_RMSE))
            print("Test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                       SROCC,
                                                                                                       PLCC,
                                                                                                       KROCC,
                                                                                         RMSE))
            torch.save(model.state_dict(), save_model)
            best_SROCC = val_SROCC

        # final test
    model.load_state_dict(torch.load(save_model))
    model.eval()
    with torch.no_grad():
        y_pred = np.zeros(testnum)
        y_test = np.zeros(testnum)
        L = 0
        for i, (patches, (label, features)) in enumerate(test_loader):
            y_test[i] = label.item()
            patches = patches.to(device)
            label = label.to(device)

            outputs = model(patches)[0]
            score = outputs.mean()

            y_pred[i] = score
            loss = criterion(score, label[0])
            L = L + loss.item()
    test_loss = L / (i + 1)
    SROCC = stats.spearmanr(y_pred, y_test)[0]
    PLCC = stats.pearsonr(y_pred, y_test)[0]
    KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
    RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())

    print("Final test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                     SROCC,
                                                                                                     PLCC,
                                                                                                     KROCC,
                                                                                                     RMSE))



















