import numpy as np
import pandas as pd
from functools import partial

import torch
import sklearn
import dgl
import errno
import json
import os

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from dgl.data.utils import Subset
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph, EarlyStopping

from models import LocalDiff
from dataset import DiffDataset

def init_featurizer(args):
    args['node_featurizer'] = CanonicalAtomFeaturizer()
    args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
    return args

def get_configure(args):
    with open(args['config_path'], 'r') as f:
        config = json.load(f)
    config['in_node_feats'] = args['node_featurizer'].feat_size()
    config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    return config

def mkdir_p(path):
    try:
        os.makedirs(path)
        print('Created directory %s'% path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory %s already exists.' % path)
        else:
            raise

def load_dataloader(args):
    train_set = DiffDataset('train', 
                        smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'])
    val_set = DiffDataset('valid', 
                        smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'])

    train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    return train_loader, val_loader

def load_model(args):
    exp_config = get_configure(args)
    model = LocalDiff(
        node_in_feats=exp_config['in_node_feats'],
        edge_in_feats=exp_config['in_edge_feats'],
        node_out_feats=exp_config['node_out_feats'],
        edge_hidden_feats=exp_config['edge_hidden_feats'],
        num_step_message_passing=exp_config['num_step_message_passing'])
    model = model.to(args['device'])
    print ('Parameters of loaded LocalDiff:')
    print (exp_config)

    if args['mode'] == 'train':
        loss_criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args['schedule_step'])
        
        if os.path.exists(args['model_path']):
            user_answer = input('%s exists, want to (a) overlap (b) continue from checkpoint (c) make a new model?' % args['model_path'])
            if user_answer == 'a':
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                print ('Overlap exsited model and training a new model...')
            elif user_answer == 'b':
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                stopper.load_checkpoint(model)
                print ('Train from exsited model checkpoint...')
            elif user_answer == 'c':
                model_name = input('Enter new model name: ')
                args['model_path'] = args['model_path'].replace('.pth', '_%s.pth' % model_name)
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                print ('Training a new model %s.pth' % model_name)
        else:
            stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
        return model, loss_criterion, optimizer, scheduler, stopper
    
    else:
        model.load_state_dict(torch.load(args['model_path'], map_location=torch.device(args['device']))['model_state_dict'])
        return model

def batch_p2r(rgraphs, pgraphs, p2r_list):
    p2rs = {}
    ratom_cnt, patom_cnt = 0, 0
    for rgraph, pgraph, p2r in zip(rgraphs, pgraphs, p2r_list):
        for patom, ratom in p2r.items():
            p2rs[patom_cnt+patom] = ratom_cnt+ratom
        ratom_cnt += rgraph.number_of_nodes()
        patom_cnt += pgraph.number_of_nodes()
    return p2rs

def collate_molgraphs(data):
    rgraphs, pgraphs, p2r_list, labels = map(list, zip(*data))
    p2rs = batch_p2r(rgraphs, pgraphs, p2r_list)
    rbg, pbg = dgl.batch(rgraphs), dgl.batch(pgraphs)
    rbg.set_n_initializer(dgl.init.zero_initializer)
    rbg.set_e_initializer(dgl.init.zero_initializer)
    pbg.set_n_initializer(dgl.init.zero_initializer)
    pbg.set_e_initializer(dgl.init.zero_initializer)
    return rbg, pbg, p2rs, torch.FloatTensor(labels)

def predict(args, model, rbg, pbg, p2rs):
    rbg = rbg.to(args['device'])
    pbg = pbg.to(args['device'])
    rnode_feats = rbg.ndata.pop('h').to(args['device'])
    redge_feats = rbg.edata.pop('e').to(args['device'])
    pnode_feats = pbg.ndata.pop('h').to(args['device'])
    pedge_feats = pbg.edata.pop('e').to(args['device'])
    return model((rbg, pbg), (rnode_feats, pnode_feats), (redge_feats, pedge_feats), p2rs)
