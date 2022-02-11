import matplotlib as mpl
import os
import sys
import pickle
import platform
import argparse
import datetime
import numpy as np
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG
from sklearn.model_selection import train_test_split

import torch
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))  # noqa
sys.path.append(os.path.join(os.path.dirname(__file__), 'layer'))  # noqa
sys.path.append(os.path.join(os.path.dirname(__file__), 'util'))  # noqa
from model_gnn_mlp import GNN  # noqa
import inv_util_dataloader as util_dataloader  # noqa

mpl.use('Agg')

torch.autograd.set_detect_anomaly(True)


def adj2lap(A):
    A = A + torch.eye(A.shape[0])
    D_sqrt_inv = torch.diag(1 / torch.sqrt(A.sum(dim=1)))
    A = D_sqrt_inv @ A @ D_sqrt_inv
    return A


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--expid', type=int, default=0)
    parser.add_argument('--guide', type=int, default=4)
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--traintest', type=str, default='rand_50')
    parser.add_argument('--trainprop', type=float, default=0.7)

    parser.add_argument('--model', type=str,
                        default="Hist2Cum-GCN-MLP-demo")
    parser.add_argument('--alpha', type=float, default=0.0,
                        help="alpha=0 mean, \inf max")

    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--hsic', type=float, default=0.0)
    parser.add_argument('--mmd', type=float, default=0.0)
    parser.add_argument('--hist2cum', type=str, default='True')

    parser.add_argument('--din', type=int, default=1)
    parser.add_argument('--dtreat', type=int, default=9)
    parser.add_argument('--dout', type=int, default=1290)

    parser.add_argument('--rep_hidden', type=str, default='40,40')  
    parser.add_argument('--out_hidden', type=str, default='400,400') 

    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step', type=int, default=50)
    parser.add_argument('--steprate', type=float, default=0.5)
    parser.add_argument('--dp', type=float, default=0.2)
    parser.add_argument('--act', type=str, default='selu')

    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()

    args.rep_hidden = [int(x) for x in args.rep_hidden.split(',')]
    args.out_hidden = [int(x) for x in args.out_hidden.split(',')]
    args.hist2cum = bool(args.hist2cum == 'True')

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    np.random.seed(1)
    torch.manual_seed(1)
    # -------------------------------- #
    dirpath = './data/'
    traintestpath = './data/dataset_%d/traintest_%s/' % (
        args.expid, args.traintest)


    _savepath = '%s/dataset_%d/guide%d/out/' % (
        dirpath, args.expid, args.guide)
    if not os.path.exists(_savepath):
        os.mkdir(_savepath)
    savepath = '%s/dataset_%d/guide%d/out/%s_a_%.1f/' % (
        dirpath, args.expid, args.guide, args.traintest, args.a)
    args.savepath = savepath
    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)

    for i in ['logs']:  
        path = savepath + i + '/'
        if not os.path.exists(path):
            os.mkdir(path)
    # -------------------------------- #

    # -------------------------------- ##
    # pytorch logger
    logname = '{}_LR_{}_BATCH_{}_{}'.format(args.model, args.lr,
                                            args.batch, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
    # , os.uname()[1]
    writer = SummaryWriter(log_dir=savepath + 'runs/' + logname)

    # -------------------------------- ##
    # Logger
    logger = getLogger("Pytorch")
    logger.setLevel(DEBUG)
    handler_format = Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(handler_format)

    file_handler = FileHandler(
        writer.log_dir + '/' + logname + '.log', 'a')

    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.debug("Start training.")
    logger.debug("SummaryWriter outputs to %s" % (writer.log_dir))
    args.log_dir = writer.log_dir
    # -------------------------------- #

    _train_id = np.loadtxt('%s/prop_train_id.csv' % (traintestpath))
    test_id = np.loadtxt('%s/prop_test_id.csv' % (traintestpath))
    train_id, valid_id = train_test_split(
        _train_id, random_state=123, test_size=1-args.trainprop)

    train_dataset = util_dataloader.ShinkokuDataset(
        id=train_id, Nguide=args.guide, a=args.a, mode='train', expid=args.expid)
    A = train_dataset.graph.get()
    W = train_dataset.getseatgraph.get_graph()

    A = adj2lap(A)
    for key in W.keys():
        W[key] = adj2lap(W[key])

    y_scaler = pickle.load(open(dirpath+'data/y_scaler.pkl', 'rb'))
    model = GNN(args.din, args.dtreat, args.dout,
                A, W, y_scaler, writer, args).to(device=args.device)

    valid_dataset = util_dataloader.ShinkokuDataset(
        id=valid_id, Nguide=args.guide, a=args.a, mode='valid', expid=args.expid)
    test_dataset_cs = util_dataloader.ShinkokuDataset(
        id=test_id, Nguide=args.guide, a=args.a, mode='train', expid=args.expid)

    in_dataset = util_dataloader.ShinkokuDataset(
        id=train_id, Nguide=args.guide, a=args.a, mode='test', expid=args.expid)
    out_dataset = util_dataloader.ShinkokuDataset(
        id=test_id, Nguide=args.guide, a=args.a, mode='test', expid=args.expid)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=False)
    validloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=False)
    testloader_cs = torch.utils.data.DataLoader(
        test_dataset_cs, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=False)
    inloader = torch.utils.data.DataLoader(
        in_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
    outloader = torch.utils.data.DataLoader(
        out_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    if args.device == 'cuda':
        model = torch.nn.DataParallel(model)  # make parallel
        torch.cudnn.benchmark = True
    logger.debug('Model Structure.')
    logger.debug(args)
    logger.debug(model)
    losses = model.fit(
        trainloader, validloader, inloader, outloader, testloader_cs)

    writer.close()
    logger.debug(0)
