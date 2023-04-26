import os
import os.path as osp
import time
import random
import warnings
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from loss import *
import lr_schedule
from logger import Logger
import pre_process as prep
from network import resnet50, resnet101
from data_list import data_batch, ImageList, SCPList

cudnn.benchmark = True
cudnn.deterministic = True
warnings.filterwarnings('ignore')


def image_classification_test(loader, base_net, classifier_layer, test_10crop, config, num_iter):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [next(iter_test[j]) for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    features_base = base_net(inputs[j])
                    output = classifier_layer(features_base)
                    outputs.append(nn.Softmax(dim=1)(output))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = classifier_layer(base_net(inputs))

                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float().cpu()), 0)           
    _, predict = torch.max(all_output, 1)
    # accuracy for each class
    class_num = config['network']['params']['class_num']
    class_accuracy = {}
    for i in range(class_num):
        class_index = all_label == i
        class_accuracy['cls_'+str(i)] = (torch.sum(torch.squeeze(predict[class_index]).float() == all_label[class_index]).item()\
            / float(all_label[class_index].size()[0])) * 100.0
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if config['is_writer']:
        class_accuracy.update({'test error': 1.0 - accuracy, 'acc': accuracy * 100.0})
        config['writer'].add_scalars('test', class_accuracy, num_iter)
    
    return accuracy * 100.0, class_accuracy


def train(config):
    # pre-process training and test data
    prep_dict = {'source': prep.image_train(**config['prep']['params']),
                 'target': prep.image_train(**config['prep']['params'])}
    if config['prep']['test_10crop']:
        prep_dict['test'] = prep.image_test_10crop(**config['prep']['params'])
    else:
        prep_dict['test'] = prep.image_test(**config['prep']['params'])

    data_set = {}
    dset_loaders = {}
    data_config = config['data']  
    
    data_set['source'] = ImageList(open(data_config['source']['list_path']).readlines(),
                                    transform=prep_dict['source'], is_train=True)  
    data_set['target'] = ImageList(open(data_config['target']['list_path']).readlines(),
                                   transform=prep_dict['target'], is_train=True)
        
    # random batch sampler or class-balance sampler for source dataloader
    if config['data']['sampler'] == 'random':
        dataloader_dict = {'batch_size': data_config['source']['batch_size'],
                           'shuffle': True, 'drop_last': True, 'num_workers': 16}
    elif config['data']['sampler'] == 'cls_balance':
        s_gt = np.array(data_set['source'].imgs)[:, 1]
        b_sampler = data_batch(s_gt, batch_size=data_config['source']['batch_size'], 
                                drop_last=False, gt_flag=True, 
                                num_class=config['network']['params']['class_num'], 
                                num_batch=data_config['source']['batch_size'])
        dataloader_dict = {'batch_sampler' : b_sampler, 'num_workers': 16}
    elif 'scp' in config['data']['sampler']:
        data_set['source'] = SCPList(open(data_config['source']['list_path']).readlines(),
                                   transform=prep_dict['source'],
                                   dynamic_p=config['data']['dynamic_p'],
                                   max_temp=config['data']['max_temp'],
                                   min_temp=config['data']['min_temp'])
        dataloader_dict = {'batch_size': data_config['source']['batch_size'],
                           'shuffle': True, 'drop_last': True, 'num_workers': 16}          
    else:
        raise ValueError('sampler %s not found!' % (config['data']['sampler'])) 
    
    dset_loaders['source'] = torch.utils.data.DataLoader(data_set['source'], 
                                                         **dataloader_dict)
    dset_loaders['target'] = torch.utils.data.DataLoader(data_set['target'],
                                                         batch_size=data_config['target']['batch_size'],
                                                         shuffle=True, num_workers=16, drop_last=True)
    if config['prep']['test_10crop']:
        data_set['test'] = [ImageList(open(data_config['test']['list_path']).readlines(),
                                      transform=prep_dict['test'][i]) for i in range(10)]
        dset_loaders['test'] = [torch.utils.data.DataLoader(dset, batch_size=data_config['test']['batch_size'],
                                                            shuffle=False, num_workers=16) for dset in data_set['test']]
    else:
        data_set['test'] = ImageList(open(data_config['test']['list_path']).readlines(), transform=prep_dict['test'])
        dset_loaders['test'] = torch.utils.data.DataLoader(data_set['test'],
                                                           batch_size=data_config['test']['batch_size'],
                                                           shuffle=False, num_workers=16)

    # set base network, classifier network, residual net
    class_num = config['network']['params']['class_num']
    net_config = config['network']
    if net_config['name'] == '50':
        base_network = resnet50()
    elif net_config['name'] == '101':
        base_network = resnet101()
    else:
        raise ValueError('base network %s not found!' % (net_config['name']))
    base_network = base_network.cuda()
    classifier_layer = nn.Linear(base_network.out_features, class_num)
    classifier_layer = classifier_layer.cuda()
    classifier_layer.weight.data.normal_(0, 0.01)
    classifier_layer.bias.data.fill_(0.0)
    softmax_layer = nn.Softmax().cuda()
    
    # set optimizer
    parameter_list = [
        {'params': base_network.parameters(), 'lr_mult': 1, 'decay_mult': 2},
        {'params': classifier_layer.parameters(), 'lr_mult': 10, 'decay_mult': 2}
    ]
    optimizer_config = config['optimizer']
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    schedule_param = optimizer_config['lr_param']
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config['lr_type']]

    # set loss
    class_criterion = nn.CrossEntropyLoss().cuda()
    loss_config = config['loss']
    if 'params' not in loss_config:
        loss_config['params'] = {}

    # class statistic
    current_cls_sta = {}
    all_cls_sta = {}

    # train
    len_train_source = len(dset_loaders['source'])
    len_train_target = len(dset_loaders['target'])
    iter_source = iter(dset_loaders['source'])
    iter_target = iter(dset_loaders['target'])
    best_acc = 0.0
    since = time.time()
    for num_iter in tqdm(range(config['max_iter'])):
        if num_iter % config['val_iter'] == 0 and num_iter != 0:
            base_network.train(False)
            classifier_layer.train(False)
            temp_acc, class_acc = image_classification_test(loader=dset_loaders, base_net=base_network,
                                                 classifier_layer=classifier_layer,
                                                 test_10crop=config['prep']['test_10crop'],
                                                 config=config, num_iter=num_iter
                                                 )
            
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = {'base': base_network.state_dict(), 'classifier': classifier_layer.state_dict()}
            log_str = 'iter: {:d}, all_accu: {:.4f},\ttime: {:.4f}'.format(num_iter, temp_acc, time.time() - since)
            config['logger'].logger.debug(log_str)
            config['results'][num_iter].append(temp_acc)
            class_acc_str = ''
            for key in class_acc.keys():
                class_acc_str += "{}: {:.4f}, ".format(key, class_acc[key])
            config['logger'].logger.debug(class_acc_str)
            class_sta_str = ''
            for i in range(class_num):
                class_sta_str += "num_cls{}: {:.4f}, ".format(str(i), current_cls_sta[str(i)])
                if str(i) in all_cls_sta:
                    all_cls_sta[str(i)] += current_cls_sta[str(i)]
                else:
                    all_cls_sta[str(i)] = current_cls_sta[str(i)]
            config['logger'].logger.debug(class_sta_str)
            if config['is_writer']:
                config['writer'].add_scalars('statistic_num', current_cls_sta, num_iter)
            current_cls_sta = {}
        # This has any effect only on modules such as Dropout or BatchNorm.
        base_network.train(True)
        classifier_layer.train(True)

        # freeze BN layers
        for m in base_network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.training = False
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        # load data
        if config['data']['sampler'] != 'cls_balance' and num_iter!=0 and num_iter % len_train_source == 0:
            iter_source = iter(dset_loaders['source'])
        if num_iter!=0 and num_iter % len_train_target == 0:
            iter_target = iter(dset_loaders['target'])
        inputs_source, labels_source = next(iter_source)
        for i in labels_source:
            if str(int(i)) in current_cls_sta:
                current_cls_sta[str(int(i))] += 1
            else:
                current_cls_sta[str(int(i))] = 1
        if config['data']['sampler'] == 'scp_d':
            dset_loaders['source'].dataset.update_current_iter()
        inputs_target, _ = next(iter_target)
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        # update lr and optimizer
        optimizer = lr_scheduler(optimizer, num_iter / config['max_iter'], **schedule_param)
        optimizer.zero_grad()

        # network forward
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features_base = base_network(inputs)
        batch_size = data_config['test']['batch_size']
        
        # source classification loss
        output_base = classifier_layer(features_base)
        classifier_loss = class_criterion(output_base[:batch_size, :], labels_source)

        # target entropy loss
        softmax_output_target = softmax_layer(output_base[batch_size:, :])
        entropy_loss = EntropyLoss(softmax_output_target)

        # alignment of L task-specific feature layers (Here, we have one layer)
        transfer_loss = MMD(features_base[:batch_size, :],
                            features_base[batch_size:, :])

        # total loss and update network
        total_loss = classifier_loss + \
                     config['loss']['alpha_off'] * transfer_loss + \
                     config['loss']['beta_off'] * entropy_loss
        total_loss.backward()
        optimizer.step()

        if num_iter % config['val_iter'] == 0:
            backbone_lr = optimizer.param_groups[0]['lr']
            classifier_lr = optimizer.param_groups[1]['lr']
            config['logger'].logger.debug(
                'class: {:.4f}\tdis: {:.4f}\tentropy: {:.4f}'.format(classifier_loss.item(), 
                                                                     config['loss']['alpha_off'] * transfer_loss,
                                                                     config['loss']['beta_off'] * entropy_loss.item()))
            config['logger'].logger.debug(
                'backbone_lr: {:.4f}\tclassifier_lr: {:.4f}'.format(backbone_lr, classifier_lr))
            if config['is_writer']:
                config['writer'].add_scalars('train', {'class': classifier_loss.item(), 'dis': transfer_loss.item(),
                                                       'entropy': config['loss']['beta_off'] * entropy_loss.item(),
                                                       'backbone_lr': backbone_lr, 'classifier_lr': classifier_lr},
                                             num_iter)
    
    class_sta_str = ''
    for i in range(class_num):
        class_sta_str += "total_num_cls{}: {:.4f}, ".format(str(i), all_cls_sta[str(i)])
    config['logger'].logger.debug(class_sta_str)
    if config['is_writer']:
        config['writer'].close()
    torch.save(best_model, osp.join(config['path']['model'], config['task'] + '_best_model.pth'))
    return best_acc


def empty_dict(config):
    config['results'] = {}
    for i in range(config['max_iter'] // config['val_iter'] + 1):
        key = config['val_iter'] * i
        config['results'][key] = []
    config['results']['best'] = []


def print_dict(config):
    for i in range(config['max_iter'] // config['val_iter'] + 1):
        key = config['val_iter'] * i
        log_str = 'setting: {:d}, average: {:.4f}'.format(key, np.average(config['results'][key]))
        config['logger'].logger.debug(log_str)
    log_str = 'best, average: {:.4f}'.format(np.average(config['results']['best']))
    config['logger'].logger.debug(log_str)
    config['logger'].logger.debug('-' * 100)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Distance based DA paradigm')
    parser.add_argument('--seed', type=int, default=1, help='manual seed')
    parser.add_argument('--gpu', type=str, nargs='?', default='2', help='device id to run')
    parser.add_argument('--net', type=str, default='50', choices=['50', '101'])
    parser.add_argument('--sampler', type=str, default='scp_d', choices=['cls_balance', 'random', 'scp', 'scp_d'])
    parser.add_argument('--data_set', default='office_cida', choices=['home', 'domainnet', 'office', 'office_cida','office_20'], help='data set')
    parser.add_argument('--source_path', type=str, default='/home/huiying/uda/dcan/data/list/office_cida/dslr_10_oneshot_addAmazon.txt', help='The source list')
    parser.add_argument('--target_path', type=str, default='/home/huiying/uda/dcan/data/list/office_cida/amazon_20_oneshot.txt', help='The target list')
    parser.add_argument('--test_path', type=str, default='/home/huiying/uda/dcan/data/list/office_cida/amazon_20_oneshot.txt', help='The test list')
    parser.add_argument('--output_path', type=str, default='snapshot/dca_dis_scp100d_da/', help='save ``log/scalar/model`` file path')
    parser.add_argument('--task', type=str, default='da', help='transfer task name')
    parser.add_argument('--max_iter', type=int, default=20001, help='max iterations')
    parser.add_argument('--val_iter', type=int, default=500, help='interval of two continuous test phase')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (Default:1e-4')
    parser.add_argument('--batch_size', type=int, default=36, help='mini batch size')
    parser.add_argument('--beta_off', type=float, default=0.1, help='target entropy loss weight ')
    parser.add_argument('--alpha_off', type=float, default=1.5, help='discrepancy loss weight')
    parser.add_argument('--is_writer', action='store_true', help='If added to sh, record for tensorboard')
    parser.add_argument('--max_temp', type=float, default=100.0, help='the value of max temperature')
    parser.add_argument('--min_temp', type=float, default=1.0, help='the value of min temperature')
    args = parser.parse_args()

    os.environ['PYTHONASHSEED'] = str(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # seed for everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    config = {'seed': args.seed, 'gpu': args.gpu, 'max_iter': args.max_iter, 'val_iter': args.val_iter,
              'data_set': args.data_set, 'task': args.task,
              'prep': {'test_10crop': True, 'params': {'resize_size': 256, 'crop_size': 224}},
              'network': {'name': args.net, 'params': {'resnet_name': args.net, 'class_num': 65}},
              'optimizer': {'type': optim.SGD,
                            'optim_params': {'lr': args.lr, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
                            'lr_type': 'inv', 'lr_param': {'lr': args.lr, 'gamma': 1.0, 'power': 0.75}},
              'data': {
                  'source': {'list_path': args.source_path, 'batch_size': args.batch_size},
                  'target': {'list_path': args.target_path, 'batch_size': args.batch_size},
                  'test': {'list_path': args.test_path, 'batch_size': args.batch_size},
                  'sampler': args.sampler},
              'output_path': args.output_path + args.data_set,
              'path': {'log': args.output_path + args.data_set + '/log/',
                       'scalar': args.output_path + args.data_set + '/scalar/',
                       'model': args.output_path + args.data_set + '/model/'},
              'is_writer': args.is_writer,
              'loss': {'alpha_off': args.alpha_off,
                      'beta_off': args.beta_off}
              }
    
    if config['data_set'] == 'home':
        config['network']['params']['class_num'] = 65
    elif config['data_set'] == 'domainnet':
        config['network']['params']['class_num'] = 345
    elif config['data_set'] == 'office':
        config['network']['params']['class_num'] = 31
    elif config['data_set'] == 'office_cida' or config['data_set'] == 'office_20':
        config['network']['params']['class_num'] = 20
    else:
        raise ValueError('dataset %s not found!' % (config['data_set']))

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
        os.makedirs(config['path']['log'])
        os.makedirs(config['path']['scalar'])
        os.makedirs(config['path']['model'])
        
    if 'scp' in config['data']['sampler']:
        config['data']['max_temp'] = args.max_temp
        config['data']['min_temp'] = args.min_temp
        config['data']['dynamic_p'] = True if 'd' in config['data']['sampler'] else False
    
    if config['is_writer']:
        config['writer'] = SummaryWriter(log_dir=config['path']['scalar'])
    config['logger'] = Logger(logroot=config['path']['log'], filename=config['task'], level='debug')
    config['logger'].logger.debug(str(config))
        
    empty_dict(config)
    config['results']['best'].append(train(config))
    print_dict(config)

