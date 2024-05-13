'''Core functionality for train/test with pytorch.'''

import locale
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pylib.utils.utils import print_time

from scripts.argparse_utils import argparse_setup, save_args
from scripts.clean_directories import clean_directories
from scripts.neural_nets.networks import get_model
from scripts.neural_nets.utils import (get_data_loaders, get_dataset,
                                       optimizer_to)
from scripts.plotting_utils import plotting_script

locale.setlocale(locale.LC_ALL, '')
torch.autograd.set_detect_anomaly(True)


def main():
    opt = argparse_setup()
    model = get_model(opt)

    core_test_train(model, opt)

def core_test_train(model, opt):
    print(opt, end = '\n\n', file = opt.log_file)
    print(opt)
    print(opt.ofile_folder, end = '\n\n')
    save_args(opt)

    # split dataset
    dataset = get_dataset(opt)
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(dataset, opt)
    val_samples = []
    for i, sample in enumerate(val_dataloader):
        sample_id = int(osp.split(sample.path[0])[1][6:])
        val_samples.append(sample_id)
        if i > 100:
            break
    print(f'First 100 val samples: {val_samples}', file = opt.log_file)

    train_loss_arr = []
    val_loss_arr = []
    if opt.resume_training:
        model_name = osp.join(opt.ofile_folder, 'model.pt')
        if osp.exists(model_name):
            if opt.cuda:
                save_dict = torch.load(model_name)
            else:
                save_dict = torch.load(model_name, map_location = 'cpu')

            model.load_state_dict(save_dict['model_state_dict'])

            opt.start_epoch = save_dict['epoch'] + 1
            print('Partially-trained model is loaded.\n'
                    f'Starting at epoch {opt.start_epoch}', file = opt.log_file)

            train_loss_arr = save_dict['train_loss']
            val_loss_arr = save_dict['val_loss']
        else:
            raise Exception(f'save_dict does not exist for {opt.ofile_folder}')
    elif opt.pretrain_id is not None:
        folder = osp.split(opt.ofile_folder)[0]
        model_name = osp.join(folder, str(opt.pretrain_id), 'model.pt')
        if osp.exists(model_name):
            if opt.cuda:
                save_dict = torch.load(model_name)
            else:
                save_dict = torch.load(model_name, map_location = 'cpu')

            model.load_state_dict(save_dict['model_state_dict'])
            print('Pre-trained model is loaded.', file = opt.log_file)
        else:
            raise Exception(f'save_dict does not exist for {model_name}')

    # Set up model and scheduler
    opt.optimizer = optim.Adam(model.parameters(), lr = opt.lr, weight_decay = opt.weight_decay)
    print(f'Optimizer: {opt.optimizer}', file = opt.log_file)
    if opt.resume_training:
        opt.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        optimizer_to(opt.optimizer, opt.device)


    if isinstance(opt.scheduler, str):
        scheduler = str(opt.scheduler).lower()
        if scheduler == 'multisteplr':
            assert opt.milestones is not None, "milestones needed for MultiStepLR"
            opt.scheduler = optim.lr_scheduler.MultiStepLR(opt.optimizer, milestones = opt.milestones,
                                                        gamma = opt.gamma, verbose = opt.verbose)

        elif scheduler == 'reducelronplateau':
            opt.scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt.optimizer, mode = 'min',
                                                    factor = opt.gamma, patience = opt.patience,
                                                    min_lr = opt.min_lr*opt.gamma)
        else:
            opt.scheduler = None
    print(f'Scheduler: {opt.scheduler}', file = opt.log_file)

    if opt.resume_training and opt.scheduler is not None:
        opt.scheduler.load_state_dict(save_dict['scheduler_state_dict'])

    if opt.use_parallel:
        model = torch.nn.DataParallel(model, device_ids = opt.gpu_ids)

    if opt.cuda:
        model.to(opt.device)

    if opt.print_params:
        print('#### INITIAL PARAMETERS ####', file = opt.param_file)
        for k,p in model.named_parameters():
            try:
                print(k, p.shape, file = opt.param_file)
                print(p, '\n', file = opt.param_file)
            except Exception as e:
                print(e, file = opt.param_file)
                print(k, file = opt.param_file)
                print(p, file = opt.param_file)
        print('\n', file = opt.param_file)

    t0 = time.time()
    print("#### TRAINING/VALIDATION ####", file = opt.log_file)
    train(train_dataloader, val_dataloader, model, opt, train_loss_arr, val_loss_arr)

    tot_pars = 0
    if opt.print_params:
        print('#### FINAL PARAMETERS ####', file = opt.param_file)
    for k,p in model.named_parameters():
        try:
            tot_pars += p.numel()
            if opt.verbose:
                print(k, p.numel(), p.shape, file = opt.log_file)
                print(p, '\ngrad: ', p.grad, '\n', file = opt.log_file)
            if opt.print_params:
                print(k, p.numel(), p.shape, file = opt.param_file)
                print(p, '\ngrad: ', p.grad, '\n', file = opt.param_file)
        except ValueError:
            print(k, file = opt.param_file)
            print(p, file = opt.param_file)
    print(f'\nTotal parameters: {locale.format_string("%d", tot_pars, grouping = True)}',
            file = opt.log_file)
    tot_time = np.round(time.time() - t0, 1)
    tot_hours = tot_time // 3600
    tot_mins = tot_time // 60
    mins = tot_mins - tot_hours*60
    secs = tot_time - tot_mins*60

    print(f'Total training + validation time: {tot_hours} hours, {mins} mins, and {secs} secs',
            file = opt.log_file)
    print(f'Final val loss: {val_loss_arr[-1]}\n', file = opt.log_file)

    if opt.GNN_mode:
        plotting_script(model, opt, train_loss_arr, val_loss_arr, dataset)
    else:
        plotting_script(model, opt, train_loss_arr, val_loss_arr)
        # don't pass dataset, need to recreate it

    # cleanup
    if opt.root is not None and opt.delete_root:
        # opt.root is set in utils.get_dataset
        clean_directories(GNN_path = opt.root, ofile = opt.log_file)
    opt.log_file.close()

def train(train_loader, val_dataloader, model, opt, train_loss = [], val_loss = []):
    def save(early_stop=False):
        if opt.use_parallel:
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        if early_stop:
            save_dict = {'model_state_dict': model_state,
                        'epoch': e,
                        'train_loss': train_loss,
                        'val_loss': val_loss}
            torch.save(save_dict, osp.join(opt.ofile_folder, 'model_early_stop.pt'))
        else:
            save_dict = {'model_state_dict': model_state,
                        'epoch': e,
                        'optimizer_state_dict': opt.optimizer.state_dict(),
                        'scheduler_state_dict': None,
                        'train_loss': train_loss,
                        'val_loss': val_loss}
            if opt.scheduler is not None:
                save_dict['scheduler_state_dict'] = opt.scheduler.state_dict()
            torch.save(save_dict, osp.join(opt.ofile_folder, 'model.pt'))

    lr = opt.lr # keep track of lr
    for e in range(opt.start_epoch, opt.n_epochs+1):
        if opt.verbose:
            print('Epoch:', e)
        model.train()
        avg_loss = 0
        for t, data in enumerate(train_loader):
            if opt.verbose:
                print('Iteration: ', t)
            opt.optimizer.zero_grad()

            data = data.to(opt.device)
            if opt.autoencoder_mode:
                y = data.contact_map
                y = torch.reshape(y, (-1, opt.m, opt.m))
            elif opt.output_mode.startswith('energy'):
                y = data.energy
                y = torch.reshape(y, (-1, opt.m, opt.m))
            else:
                y = data.y
                y = torch.reshape(y, (-1, opt.m))
            if opt.verbose:
                # print(f'x: shape={data.x.shape}, '
                #         f'min={torch.min(data.x).item()}, '
                #         f'max={torch.max(data.x).item()}')
                # if data.edge_attr is not None:
                #     print(f'edge_attr: '
                #             f'shape={data.edge_attr.shape}, '
                #             f'min={torch.min(data.edge_attr).item()}, '
                #             f'max={torch.max(data.edge_attr).item()}')
                # if 'pos_edge_attr' in data._mapping:
                #     print(f'pos_edge_attr={data.pos_edge_attr}, '
                #             f'shape={data.pos_edge_attr.shape}, '
                #             f'min={torch.min(data.pos_edge_attr).item()}, '
                #             f'max={torch.max(data.pos_edge_attr).item()}')
                #     print(f'neg_edge_attr={data.neg_edge_attr}, '
                #             f'shape={data.neg_edge_attr.shape}, '
                #             f'min={torch.min(data.neg_edge_attr).item()}, '
                #             f'max={torch.max(data.neg_edge_attr).item()}')
                # print(f'y: shape={y.shape}, min={torch.min(y).item()}, '
                #         f'max={torch.max(y).item()}')
                t0 = time.time()
            yhat = model(data)
            if opt.verbose:
                tf = time.time()
                print_time(t0, tf, 'forward')
                # print(f'yhat={yhat}, shape={yhat.shape}, '
                #         f'min={torch.min(yhat).item()}, '
                #         f'max={torch.max(yhat).item()}')
            if 'seqs' in data._mapping:
                seqs = torch.reshape(data.seqs, (-1, 10, opt.m)) # TODO hard-coded 10
                loss = opt.criterion(yhat, y, seqs)
            else:
                loss = opt.criterion(yhat, y)
            if opt.w_reg is not None:
                if opt.w_reg == 'l1':
                    loss += opt.reg_lambda * torch.norm(model.sym(model.W), 1)
                elif opt.w_reg == 'l2':
                    loss += opt.reg_lambda * torch.norm(model.sym(model.W), 2) ** 2
            avg_loss += loss.item()
            loss.backward()
            # if opt.verbose:
            #     for k,p in model.named_parameters():
            #         print(f'{k}: shape={p.shape}, min={torch.min(p).item()}, '
            #                 f'max={torch.max(p).item()}')
            #         print(f'\tgrad: shape={p.grad.shape}, min={torch.min(p.grad).item()}, '
            #                 f'max={torch.max(p.grad).item()}')
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            opt.optimizer.step()
        avg_loss /= (t+1)
        train_loss.append(avg_loss)

        if e % opt.print_mod == 0 or e == opt.n_epochs:
            print('Epoch {}, loss = {:.4f}'.format(e, avg_loss), file = opt.log_file)
            print_val_loss = True
            opt.log_file.close() # save any writes so far
            opt.log_file = open(opt.log_file_path, 'a')
        else:
            print_val_loss = False

        val_loss_i = test(val_dataloader, model, opt, print_val_loss)
        val_loss.append(val_loss_i)

        if opt.scheduler is not None:
            if opt.milestones is None:
                opt.scheduler.step(val_loss_i)
            else:
                opt.scheduler.step()
            new_lr = opt.scheduler.optimizer.param_groups[0]['lr']
            if new_lr < lr:
                lr = new_lr
                print(f'New lr: {lr}', file = opt.log_file)
                if opt.save_early_stop:
                    save(True)

        if opt.min_lr is not None and lr < opt.min_lr:
            print('Converged', file = opt.log_file)
            save()
            break

        if e % opt.save_mod == 0:
            save()

    return train_loss, val_loss

def test(loader, model, opt, toprint):
    assert loader is not None, 'loader is None - check train/val/test split'
    model.eval()
    loss_list = []
    with torch.no_grad():
        for t, data in enumerate(loader):
            data = data.to(opt.device)
            if opt.autoencoder_mode:
                y = data.contact_map
                y = torch.reshape(y, (-1, opt.m, opt.m))
            elif opt.output_mode.startswith('energy'):
                y = data.energy
                y = torch.reshape(y, (-1, opt.m, opt.m))
            else:
                y = data.y
                y = torch.reshape(y, (-1, opt.m))
            yhat = model(data)
            if 'seqs' in data._mapping:
                seqs = torch.reshape(data.seqs, (-1, 10, opt.m)) # TODO hard-coded 10
                loss = opt.criterion(yhat, y, seqs)
            else:
                loss = opt.criterion(yhat, y)
            loss_list.append(loss.item())

    avg_loss = np.mean(loss_list)
    if toprint:
        print(f'Mean test/val loss: {avg_loss:.4f}', file = opt.log_file)
        percentiles = np.percentile(loss_list, [25, 50, 75])
        percentiles = np.round(percentiles, 4)
        print(f'[25, 50, 75] percentiles test/val loss: {percentiles}\n', file = opt.log_file)

    return avg_loss


if __name__ == '__main__':
    main()
