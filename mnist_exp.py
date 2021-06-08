import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim
import argparse
import tensorboard_logger as tb_logger
import hypergrad as hg
import time
import math

from itertools import repeat
from torch.nn import functional as F
from torchvision import datasets
from copy import deepcopy
from stocBiO import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--spider_size', type=int, default=32)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=50, help='K')
    parser.add_argument('--iterations', type=int, default=200, help='T')
    parser.add_argument('--outer_lr', type=float, default=0.1, help='beta')
    parser.add_argument('--inner_lr', type=float, default=0.1, help='alpha')
    parser.add_argument('--eta', type=float, default=0.5, help='used in Hessian')
    parser.add_argument('--data_path', default='data/', help='The temporary data storage path')
    parser.add_argument('--training_size', type=int, default=20000)
    parser.add_argument('--validation_size', type=int, default=5000)
    parser.add_argument('--noise_rate', type=float, default=0.1)
    parser.add_argument('--hessian_q', type=int, default=3)
    parser.add_argument('--spider_epoch', type=int, default=3)
    parser.add_argument('--save_folder', type=str, default='', help='path to save result')
    parser.add_argument('--model_name', type=str, default='', help='Experiment name')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pretrain_epoch', type=int, default=5)
    parser.add_argument('--mode', type=str, default='pretrain', choices=['train', 'pretrain'])
    parser.add_argument('--alg', type=str, default='stocBiO', choices=['stocBiO', 'HOAG', 'TTSA', 'BSA', 
                                                        'reverse', 'AID-CG', 'AID-FP', 'VRBO', 'MRBO',
                                                        'MSTSA', 'STABLE'])
    args = parser.parse_args()

    if args.alg == 'stocBiO':
        args.batch_size = args.batch_size
    elif args.alg == 'STABLE':
        args.batch_size = args.batch_size
    elif args.alg == 'MRBO':
        args.batch_size = args.batch_size
    elif args.alg == 'VRBO':
        args.batch_size = args.batch_size
    elif args.alg == 'BSA':
        args.batch_size = 1
    elif args.alg == 'TTSA':
        args.batch_size = 1
        args.iterations = 1
    elif args.alg == 'MSTSA':
        args.batch_size = 1
    else:
        args.batch_size = args.training_size
    
    if not args.save_folder:
        args.save_folder = './save_tb_results'
    args.model_name = '{}_{}_bs_{}_olr_{}_ilr_{}_eta_{}_noiser_{}_q_{}_ite_{}'.format(args.alg, 
                       args.training_size, args.batch_size, args.outer_lr, args.inner_lr, args.eta, 
                       args.noise_rate, args.hessian_q, args.iterations)
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    return args

def get_data_loaders(args):
    kwargs = {'num_workers': 0, 'pin_memory': True}
    dataset = datasets.MNIST(root=args.data_path, train=True, download=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    train_sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler,
        batch_size=args.batch_size, **kwargs)
    train_spider_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler,
        batch_size=args.spider_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(root=args.data_path, train=False,
                        download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=args.test_size)
    return train_loader, test_loader, train_spider_loader


def train_model(args, train_loader, test_loader, spider_loader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.mode == 'train':
        parameters = torch.load('pretrained_{}.pt'.format(args.pretrain_epoch)).to(device)
        lambda_x = torch.load('pretrained_lambda_{}.pt'.format(args.pretrain_epoch)).to(device)
    elif args.mode == 'pretrain':
        parameters = torch.randn((args.num_classes, 785), requires_grad=True)
        parameters = nn.init.kaiming_normal_(parameters, mode='fan_out').to(device)
        lambda_x = torch.zeros((args.training_size), requires_grad=True).to(device)

    
    
    # hyperparameter: lambda_x
    loss_time_results = np.zeros((args.epochs+1, 4))
    batch_num = args.training_size//args.batch_size
    train_loss_avg = loss_train_avg(train_loader, parameters, device, batch_num)
    test_loss_avg = loss_test_avg(test_loader, parameters, device)
    loss_time_results[0, :] = [train_loss_avg, test_loss_avg, (0.0), (0.0)]
    print('Epoch: {:d} Train Loss: {:.4f} Test Loss: {:.4f}'.format(0, train_loss_avg, test_loss_avg))
    
    images_list, labels_list = [], []
    for index, (images, labels) in enumerate(train_loader):
        images_list.append(images)
        labels_list.append(labels)
    
    images_spider_list, labels_spider_list = [], []
    for index, (images, labels) in enumerate(spider_loader):
        images_spider_list.append(images)
        labels_spider_list.append(labels)

    # setting for reverse, fixed_point & CG
    def loss_inner(parameters, weight, data_all):
        data = data_all[0]
        labels = data_all[1]
        data = torch.reshape(data, (data.size()[0],-1)).to(device)
        labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
        output = torch.matmul(data, torch.t(parameters[0][:, 0:784]))+parameters[0][:, 784]
        loss = F.cross_entropy(output, labels_cp, reduction='none')
        loss_regu = torch.mean(torch.mul(loss, torch.sigmoid(weight[0]))) + 0.001*torch.pow(torch.norm(parameters[0]),2)
        return loss_regu

    def loss_outer(parameters, lambda_x):
        images, labels = images_list[-1], labels_list[-1]
        images = torch.reshape(images, (images.size()[0],-1)).to(device)
        images_temp, labels_temp = images[0:args.validation_size,:], labels[0:args.validation_size]
        images = torch.cat([images_temp]*(args.training_size // args.validation_size))
        labels = torch.cat([labels_temp]*(args.training_size // args.validation_size))
        output = torch.matmul(images, torch.t(parameters[0][:, 0:784]))+parameters[0][:, 784]
        loss = F.cross_entropy(output, labels)
        return loss

    def out_f(data, parameters):
        output = torch.matmul(data, torch.t(parameters[:, 0:784]))+parameters[:, 784]
        return output

    def reg_f(params, hparams, loss):
        loss_regu = torch.mean(torch.mul(loss, torch.sigmoid(hparams))) + 0.001*torch.pow(torch.norm(params),2)
        return loss_regu

    def vrbo(args, val_data_list, parameters, hparams, hparams_old, grad_x, grad_y): 
        data_list, labels_list = val_data_list[0]
        output = out_f(data_list[1], parameters)
        update_y = gradient_gy(args, labels_list[1], parameters, data_list[1], hparams, output, reg_f) 
        update_y_old = gradient_gy(args, labels_list[1], parameters, data_list[1], hparams_old, output, reg_f)
        update_x = stocbio(parameters, hparams, val_data_list[0], args, out_f, reg_f)
        update_x_old = stocbio(parameters, hparams_old, val_data_list[0], args, out_f, reg_f)

        v_t = grad_x + update_x - update_x_old
        u_t = grad_y + update_y - update_y_old
        parameters_new = parameters - args.inner_lr*u_t
        for t in range(args.iterations):
            data_list, labels_list = val_data_list[t+1]
            output = out_f(data_list[1], parameters_new)
            update_y = gradient_gy(args, labels_list[1], parameters_new, data_list[1], hparams, output, reg_f) 
            output = out_f(data_list[1], parameters)
            update_y_old = gradient_gy(args, labels_list[1], parameters, data_list[1], hparams, output, reg_f)
            update_x = stocbio(parameters_new, hparams, val_data_list[t+1], args, out_f, reg_f)
            update_x_old = stocbio(parameters, hparams, val_data_list[t+1], args, out_f, reg_f)
            
            v_t = v_t + update_x - update_x_old 
            u_t = u_t + update_y - update_y_old 
            parameters = parameters_new
            parameters_new = parameters - args.inner_lr*u_t
        return parameters_new, v_t, u_t


    tol = 1e-12
    warm_start = True
    params_history = []
    train_iterator = repeat([images_list[0], labels_list[0]])
    inner_opt = hg.GradientDescent(loss_inner, args.inner_lr, data_or_iter=train_iterator)
    inner_opt_cg = hg.GradientDescent(loss_inner, 1., data_or_iter=train_iterator)
    outer_opt = torch.optim.SGD(lr=args.outer_lr, params=[lambda_x])
    
    start_time = time.time() 
    c_1, c_2=1,1

    lambda_index_outer = 0
    for epoch in range(args.epochs):
        grad_norm_inner = 0.0
        if args.alg == 'stocBiO' or args.alg == 'HOAG':
            train_index_list = torch.randperm(batch_num)
            for index in range(args.iterations):
                index_rn = train_index_list[index%batch_num]
                images, labels = images_list[index_rn], labels_list[index_rn]
                images = torch.reshape(images, (images.size()[0],-1)).to(device)
                labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
                weight = lambda_x[index_rn*args.batch_size: (index_rn+1)*args.batch_size]
                output = out_f(images, parameters)
                inner_update = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f)
                parameters = parameters - args.inner_lr*inner_update
                if index == args.iterations-1:
                    grad_norm_inner = torch.norm(inner_update)
                    print("Inner update: {:.4f}".format(grad_norm_inner))

            if args.mode == 'pretrain': 
                if epoch == args.pretrain_epoch:
                    torch.save(parameters, 'pretrained_{}.pt'.format(args.pretrain_epoch))
                    torch.save(lambda_x, 'pretrained_lambda_{}.pt'.format(args.pretrain_epoch))
        
        if args.alg == 'stocBiO':
            val_index = torch.randperm(args.validation_size//args.batch_size)
            val_data_list = build_val_data(args, val_index, images_list, labels_list, device)
            hparams = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
            outer_update = stocbio(parameters, hparams, val_data_list, args, out_f, reg_f)
            print("Outer update: {:.4f}".format(torch.norm(outer_update)))
        
        elif args.alg == 'HOAG':
            images, labels = images_list[-1], labels_list[-1]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            images_temp, labels_temp = images[0:args.validation_size,:], labels[0:args.validation_size]
            images = torch.cat([images_temp]*(args.training_size // args.validation_size))
            labels = torch.cat([labels_temp]*(args.training_size // args.validation_size))

            # Fy_gradient
            labels = labels.to(device)
            output = out_f(images, parameters)
            Fy_gradient = gradient_fy(args, labels, parameters, images, output)
            v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

            # Hessian
            z_list = []
            v_Q = args.eta*v_0
            labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
            weight = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
            output = out_f(images, parameters)
            Gy_gradient = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f)
            Gy_gradient = torch.reshape(Gy_gradient, [-1])
            G_gradient = torch.reshape(parameters, [-1]) - args.eta*Gy_gradient
            for q in range(args.hessian_q):
                Jacobian = torch.matmul(G_gradient, v_0)
                v_new = torch.autograd.grad(Jacobian, parameters, retain_graph=True)[0]
                v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
                z_list.append(v_0)

            v_Q = v_Q+torch.sum(torch.stack(z_list), dim=0)
            Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), weight)[0]
            outer_update = -Gyx_gradient

        elif args.alg == 'BSA' or args.alg == 'TTSA':
            train_index_list = torch.randperm(args.training_size)
            random_list = np.random.uniform(size=[args.training_size])
            noise_rate_list = np.where((random_list>args.noise_rate), 0, 1)
            for index in range(args.iterations):
                images, labels = images_list[train_index_list[index]], labels_list[train_index_list[index]]
                images = torch.reshape(images, (images.size()[0],-1)).to(device)
                labels_cp = nositify(labels, noise_rate_list[index], args.num_classes).to(device)
                weight = lambda_x[train_index_list[index]: train_index_list[index]+1]
                output = out_f(images, parameters)
                inner_update = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f)
                parameters = parameters - args.inner_lr*inner_update

            val_index = -torch.randperm(args.validation_size)
            random_list = np.random.uniform(size=[args.hessian_q+2])
            noise_rate_list = np.where((random_list>args.noise_rate), 0, 1)

            # Fy_gradient
            images, labels = images_list[val_index[1]], labels_list[val_index[1]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels = labels.to(device)
            output = out_f(images, parameters)
            Fy_gradient = gradient_fy(args, labels, parameters, images, output)
            v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()
            weight = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]

            # Hessian
            z_list = []
            v_Q = args.eta*v_0
            images, labels = images_list[val_index[2]], labels_list[val_index[2]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = nositify(labels, noise_rate_list[1], args.num_classes).to(device)
            output = out_f(images, parameters)
            Gy_gradient = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f)
            Gy_gradient = torch.reshape(Gy_gradient, [-1])
            G_gradient = torch.reshape(parameters, [-1]) - args.eta*Gy_gradient
            for q in range(args.hessian_q):
                Jacobian = torch.matmul(G_gradient, v_0)
                v_new = torch.autograd.grad(Jacobian, parameters, retain_graph=True)[0]
                v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
                z_list.append(v_0)            
            v_Q = v_Q+torch.sum(torch.stack(z_list), dim=0)

            # Gyx_gradient
            images, labels = images_list[val_index[0]], labels_list[val_index[0]]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = nositify(labels, noise_rate_list[0], args.num_classes).to(device)
            output = out_f(images, parameters)
            Gy_gradient = gradient_gy(args, labels_cp, parameters, images, weight, output, reg_f)
            Gy_gradient = torch.reshape(Gy_gradient, [-1])
            Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), weight)[0]
            outer_update = -Gyx_gradient      

        elif args.alg == 'VRBO':
            if epoch%args.spider_epoch == 0:
                hparams = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
                val_index = torch.randperm(args.validation_size//args.batch_size)
                val_data_list = build_val_data(args, val_index, images_list, labels_list, device)
                grad_x = stocbio(parameters, hparams, val_data_list, args, out_f, reg_f)
                data_spider_list, labels_spider_list_0 = val_data_list
                output = out_f(data_spider_list[1], parameters)
                grad_y = gradient_gy(args, labels_spider_list_0[1], parameters, data_spider_list[1], hparams, output, reg_f)
                # hparams_old = hparams

                hparams = lambda_x[lambda_index_outer+args.batch_size: lambda_index_outer+args.batch_size+args.spider_size]
                hparams_old = lambda_x[lambda_index_outer-args.spider_size+args.batch_size: lambda_index_outer+args.batch_size]
                val_data_list = build_spider_data(args, images_spider_list, labels_spider_list, device)
                paras_new, grad_x, grad_y = vrbo(args, val_data_list, parameters, hparams, 
                    hparams_old, grad_x[0:args.spider_size], grad_y)
                # hparams_old = hparams
                parameters = paras_new
            else:
                hparams = lambda_x[lambda_index_outer: lambda_index_outer+args.spider_size]
                hparams_old = lambda_x[lambda_index_outer-args.spider_size: lambda_index_outer]
                val_data_list = build_spider_data(args, images_spider_list, labels_spider_list, device)
                paras_new, grad_x, grad_y = vrbo(args, val_data_list, parameters, hparams, 
                    hparams_old, grad_x[0:args.spider_size], grad_y)
                # hparams_old = hparams
                parameters = paras_new

            grad_norm_inner = torch.norm(grad_y)
            print("Inner update: {:.4f}".format(grad_norm_inner))
            print("Outer update: {:.4f}".format(torch.norm(grad_x)))
            weight = hparams
            outer_update = grad_x

        elif args.alg == 'MRBO':
            hparams = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
            # hparams_old = lambda_x[lambda_index_outer-args.batch_size: lambda_index_outer]
            val_index = -torch.randperm(args.training_size//args.batch_size)
            val_data_list = build_val_data(args, val_index, images_list, labels_list, device)
            data_s_list, labels_s_list = val_data_list
            eta_k, alpha_k, beta_k, m = 1.0, 0.9, 0.9, 0.1

            if epoch == 0:
                grad_x = stocbio(parameters, hparams, val_data_list, args, out_f, reg_f)
                output = out_f(data_s_list[1], parameters)
                grad_y = gradient_gy(args, labels_s_list[1], parameters, data_s_list[1], hparams, output, reg_f)
            
                parameters_old,grad_y_old = parameters, grad_y
                parameters = parameters - args.inner_lr*eta_k*grad_y
            else:

                # for t in range(args.iterations):
                output = out_f(data_s_list[1], parameters)
                output_old = out_f(data_s_list[1], parameters_old)
                update_y = gradient_gy(args, labels_s_list[1], parameters, data_s_list[1], hparams, output, reg_f) 
                update_y_old = gradient_gy(args, labels_s_list[1], parameters_old, data_s_list[1], hparams_old, output_old, reg_f)
                grad_y = update_y+(1-beta_k)*(grad_y_old-update_y_old)
                parameters_old,grad_y_old = parameters, grad_y
                parameters = parameters - args.inner_lr*eta_k*grad_y

                update_x = stocbio(parameters, hparams, val_data_list, args, out_f, reg_f)
                update_x_old = stocbio(parameters_old, hparams_old, val_data_list, args, out_f, reg_f)
                grad_x = update_x+(1-alpha_k)*(grad_x_old-update_x_old)
                
                
            parameters_old, hparams_old, grad_x_old, grad_y_old = parameters, hparams, grad_x, grad_y
            parameters, hparams = parameters - args.inner_lr*eta_k*grad_y, hparams - args.outer_lr*eta_k*grad_x
            outer_update = grad_x
            grad_norm_inner = torch.norm(grad_y)
            print("Inner update: {:.4f}".format(grad_norm_inner))
            print("Outer update: {:.4f}".format(torch.norm(grad_x)))
            weight = hparams
            eta_k = eta_k*(((epoch+m)/(epoch+m+1))**(1/3))
            # print(eta_k)
            alpha_k, beta_k=alpha_k*(eta_k**2), beta_k*(eta_k**2)

        elif args.alg == 'MSTSA':
            hparams = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
            if epoch == 0:
                outer_update_old = 0
                hparams_old = hparams
            else:
                hparams_old = lambda_x[lambda_index_outer-args.batch_size: lambda_index_outer]
            # args.outer_lr = 0.1
            # args.inner_lr=0.1 
            c_eta = 0.5
            args.outer_lr = 0.1/(math.sqrt(epoch+1))
            beta_t = args.inner_lr/(math.sqrt(epoch+1))

            images, labels = images_list[epoch%batch_num], labels_list[epoch%batch_num]
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)    
            output = out_f(images, parameters)
            inner_update = gradient_gy(args, labels_cp, parameters, images, hparams, output, reg_f)
            params_next = parameters - beta_t*inner_update
            
            val_index = torch.randperm(args.validation_size//args.batch_size)
            val_data_list = build_val_data(args, val_index, images_list, labels_list, device)
            outer_update = mstsa(outer_update_old, c_eta, val_data_list, args, parameters, 
                params_next, hparams, hparams_old, out_f, reg_f)
            parameters = params_next

            weight = hparams

        elif args.alg == 'STABLE':
            hparams = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]
            if epoch == 0:
                hparams_old = hparams
                params_old = parameters
                H_xy = torch.zeros([args.batch_size, 7850])
                H_yy = torch.zeros([7850, 7850])
            else:
                hparams_old = lambda_x[lambda_index_outer-args.batch_size: lambda_index_outer]
            beta_k = args.inner_lr
            alpha_k = args.outer_lr
            tao = 0.5
            
            val_index = torch.randperm(args.training_size//args.batch_size)
            data_list = build_val_data(args, -val_index, images_list, labels_list, device)
            params_old, parameters, outer_update, H_xy, H_yy = stable(args, data_list, params_old, parameters, 
                hparams, hparams_old, H_xy, H_yy, tao, beta_k, alpha_k, out_f, reg_f)
            
            # args.outer_lr = alpha_k
            weight = hparams

        else:
            inner_losses = []
            if params_history:
                params_history = [params_history[-1]]
            else:
                params_history = [[parameters]]
            for index in range(args.iterations):
                params_history.append(inner_opt(params_history[-1], [lambda_x], create_graph=False))
                inner_losses.append(inner_opt.curr_loss)

            final_params = params_history[-1]
            outer_opt.zero_grad()
            if args.alg == 'reverse':
                hg.reverse(params_history[-args.hessian_q-1:], [lambda_x], [inner_opt]*args.hessian_q, loss_outer)
            elif args.alg == 'AID-FP':
                hg.fixed_point(final_params, [lambda_x], args.hessian_q, inner_opt, loss_outer, stochastic=False, tol=tol)
            elif args.alg == 'AID-CG':
                hg.CG(final_params[:len(parameters)], [lambda_x], args.hessian_q, inner_opt_cg, loss_outer, stochastic=False, tol=tol)
            outer_update = lambda_x.grad
            weight = lambda_x[lambda_index_outer: lambda_index_outer+args.batch_size]

        outer_update = torch.squeeze(outer_update)
        with torch.no_grad():
            weight = weight - args.outer_lr*outer_update
        if (args.alg == 'VRBO') & (epoch % args.spider_epoch !=0):
            lambda_index_outer = (lambda_index_outer+args.spider_size) % args.training_size
        else:
            lambda_index_outer = (lambda_index_outer+args.batch_size) % args.training_size

        if args.alg == 'reverse' or args.alg == 'AID-CG' or args.alg == 'AID-FP':
            train_loss_avg = loss_train_avg(train_loader, final_params[0], device, batch_num)
            test_loss_avg = loss_test_avg(test_loader, final_params[0], device)
        else:
            train_loss_avg = loss_train_avg(train_loader, parameters, device, batch_num)
            test_loss_avg = loss_test_avg(test_loader, parameters, device)
        end_time = time.time()
        print('Epoch: {:d} Train Loss: {:.4f} Test Loss: {:.4f} Time: {:.4f}'.format(epoch+1, train_loss_avg, test_loss_avg,
                        (end_time-start_time)))

        loss_time_results[epoch+1, 0] = train_loss_avg
        loss_time_results[epoch+1, 1] = test_loss_avg
        loss_time_results[epoch+1, 2] = (end_time-start_time)
        loss_time_results[epoch+1, 3] = grad_norm_inner

    print(loss_time_results)
    file_name = str(args.seed)+'.npy'
    file_addr = os.path.join(args.save_folder, file_name)
    with open(file_addr, 'wb') as f:
            np.save(f, loss_time_results)

def loss_train_avg(data_loader, parameters, device, batch_num):
    loss_avg, num = 0.0, 0
    for index, (images, labels) in enumerate(data_loader):
        if index>= batch_num:
            break
        else:
            images = torch.reshape(images, (images.size()[0],-1)).to(device)
            labels = labels.to(device)
            loss = loss_f_funciton(labels, parameters, images)
            loss_avg += loss 
            num += 1
    loss_avg = loss_avg/num
    return loss_avg.detach()

def loss_test_avg(data_loader, parameters, device):
    loss_avg, num = 0.0, 0
    for _, (images, labels) in enumerate(data_loader):
        images = torch.reshape(images, (images.size()[0],-1)).to(device)
        # images = torch.cat((images, torch.ones(images.size()[0],1)),1)
        labels = labels.to(device)
        loss = loss_f_funciton(labels, parameters, images)
        loss_avg += loss 
        num += 1
    loss_avg = loss_avg/num
    return loss_avg.detach()

def loss_f_funciton(labels, parameters, data):
    output = torch.matmul(data, torch.t(parameters[:, 0:784]))+parameters[:, 784]
    loss = F.cross_entropy(output, labels)
    return loss

def nositify(labels, noise_rate, n_class):
    num = noise_rate*(labels.size()[0])
    num = int(num)
    randint = torch.randint(1, 10, (num,))
    index = torch.randperm(labels.size()[0])[:num]
    labels[index] = (labels[index]+randint) % n_class
    return labels

def build_val_data(args, val_index, images_list, labels_list, device):
    val_index = -(val_index)
    val_data_list, val_labels_list = [], []
    
    images, labels = images_list[val_index[0]], labels_list[val_index[0]]
    images = torch.reshape(images, (images.size()[0],-1)).to(device)
    labels = labels.to(device)
    val_data_list.append(images)
    val_labels_list.append(labels)

    images, labels = images_list[val_index[1]], labels_list[val_index[1]]
    images = torch.reshape(images, (images.size()[0],-1)).to(device)
    labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
    val_data_list.append(images)
    val_labels_list.append(labels_cp)

    images, labels = images_list[val_index[2]], labels_list[val_index[2]]
    images = torch.reshape(images, (images.size()[0],-1)).to(device)
    labels_cp = nositify(labels, args.noise_rate, args.num_classes).to(device)
    val_data_list.append(images)
    val_labels_list.append(labels_cp)

    return [val_data_list, val_labels_list]

def build_spider_data(args, images_spider_list, labels_spider_list, device):
    val_data_list = []
    # use training data
    for iter in range(args.iterations+1):
        spider_index = torch.randperm(args.training_size//args.spider_size)
        val_data = build_val_data(args, -spider_index, images_spider_list, labels_spider_list, device)
        val_data_list.append(val_data)
    return val_data_list

def main():
    args = parse_args()
    print(args)
    train_loader, test_loader, spider_loader = get_data_loaders(args)
    train_model(args, train_loader, test_loader, spider_loader)

if __name__ == '__main__':
    main()
