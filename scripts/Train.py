from argparse import ArgumentParser
import numpy as np
import torch
import sklearn
import torch.nn as nn

from utils import init_featurizer, mkdir_p, get_configure, load_model, load_dataloader, predict

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_id, batch_data in enumerate(data_loader):
        rbg, pbg, p2rs, labels = batch_data
        if len(labels) == 1:
            continue
        labels = labels.to(args['device'])
        preds = predict(args, model, rbg, pbg, p2rs)
        loss = loss_criterion(preds, labels)
        train_loss += loss.item()
        
        optimizer.zero_grad()      
        loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        optimizer.step()
                
        if batch_id % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, rmse loss: %.4f' % (epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), np.sqrt(loss.item())), end='', flush=True)

    print('\nepoch %d/%d, training loss: %.4f' % (epoch + 1, args['num_epochs'], np.sqrt(train_loss/batch_id)))

def run_an_eval_epoch(args, model, data_loader, loss_criterion):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            rbg, pbg, p2rs, labels = batch_data
            labels = labels.to(args['device'])
            preds = predict(args, model, rbg, pbg, p2rs)
            loss = loss_criterion(preds, labels)
            val_loss += loss.item()
    return np.sqrt((val_loss/batch_id))


def main(args):
    model_name = 'LocalDiff.pth'
    args['model_path'] = '../models/' + model_name
    args['config_path'] = '../data/configs/%s' % args['config']
    mkdir_p('../models')                          
    args = init_featurizer(args)
    model, loss_criterion, optimizer, scheduler, stopper = load_model(args)   
    train_loader, val_loader = load_dataloader(args)
    for epoch in range(args['num_epochs']):
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        val_loss = run_an_eval_epoch(args, model, val_loader, loss_criterion)
        early_stop = stopper.step(val_loss, model) 
        scheduler.step()
        print('epoch %d/%d, validation RMSE loss: %.4f' %  (epoch + 1, args['num_epochs'], val_loss))
        print('epoch %d/%d, Best loss: %.4f' % (epoch + 1, args['num_epochs'], stopper.best_score))
        if early_stop:
            print ('Early stopped!!')
            break
    return
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-g', '--gpu', default='cpu', help='GPU device to use')
    parser.add_argument('-c', '--config', default='default_config.json', help='Configuration of model')
    parser.add_argument('-b', '--batch-size', default=12, help='Batch size of dataloader')                             
    parser.add_argument('-n', '--num-epochs', type=int, default=50, help='Maximum number of epochs for training')
    parser.add_argument('-p', '--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float, default=1e-6, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int, default=20, help='Step size of learning scheduler')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    parser.add_argument('-pe', '--print-every', type=int, default=20, help='Print the training progress every X mini-batches')
    args = parser.parse_args().__dict__
    args['mode'] = 'train'
    print (args)
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print ('Using device %s' % args['device'])
    main(args)