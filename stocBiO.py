import torch
from torch.autograd import grad 
from torch.nn import functional as F

def mstsa(prerious_update, eta, val_data_list, args, params, params_next, hparams, hparams_old, out_f, reg_f):
    grad1 = stocbio(params_next, hparams, val_data_list, args, out_f, reg_f)
    grad2 = stocbio(params, hparams_old, val_data_list, args, out_f, reg_f)
    outer_update = eta*grad1+(1-eta)*(prerious_update+grad1-grad2)
    return outer_update

def stable(args, train_data_list, params_old, params, hparams, hparams_old, H_xy_old, H_yy_old, 
        tao, beta_k, alpha_k, out_f, reg_f):
    data_list, labels_list = train_data_list

    output = out_f(data_list[0], params)
    grad_fy = gradient_fy(args, labels_list[0], params, data_list[0], output)
    grad_fy = torch.reshape(grad_fy, [-1])
    output = out_f(data_list[1], params)
    grad_gy = gradient_gy(args, labels_list[1], params, data_list[1], hparams, output, reg_f)
    grad_gy = torch.reshape(grad_gy, [-1])
    output = out_f(data_list[2], params_old)
    grad_gy_old = gradient_gy(args, labels_list[2], params_old, data_list[2], hparams_old, output, reg_f)
    grad_gy_old = torch.reshape(grad_gy_old, [-1])

    h_xy_k0, h_xy_k1, h_yy_k0, h_yy_k1 = [], [], [], []
    for index in range(grad_gy.size()[0]):
        h_xy_k0.append(torch.autograd.grad(grad_gy_old[index], hparams_old, retain_graph=True)[0])
        h_xy_k1.append(torch.autograd.grad(grad_gy[index], hparams, retain_graph=True)[0])
        h_yy_k0.append(torch.autograd.grad(grad_gy_old[index], params_old, retain_graph=True)[0])
        h_yy_k1.append(torch.autograd.grad(grad_gy[index], params, retain_graph=True)[0])
    h_xy_k0,h_xy_k1,h_yy_k0,h_yy_k1 = torch.stack(h_xy_k0), torch.stack(h_xy_k1), torch.stack(h_yy_k0),torch.stack(h_yy_k1)
    h_yy_k0 = torch.reshape(h_yy_k0, [7850,-1])
    h_yy_k1 = torch.reshape(h_yy_k1, [7850,-1])

    H_xy = (1-tao)*(H_xy_old-torch.t(h_xy_k0))+torch.t(h_xy_k1)
    H_yy = (1-tao)*(H_yy_old-torch.t(h_yy_k0))+torch.t(h_yy_k1)
    
    x_update = -torch.matmul(torch.matmul(H_xy, torch.inverse(H_yy)), grad_fy)
    params_shape = params.size()
    temp = torch.matmul(torch.matmul(torch.inverse(H_yy),torch.t(H_xy)),(-x_update*alpha_k))
    params_new = torch.reshape(params, [-1]) - beta_k*grad_gy-temp
    params_new = torch.reshape(params_new, params_shape)
    return params, params_new, x_update, H_xy, H_yy

def stocbio(params, hparams, val_data_list, args, out_f, reg_f):
        data_list, labels_list = val_data_list
        # Fy_gradient
        output = out_f(data_list[0], params)
        Fy_gradient = gradient_fy(args, labels_list[0], params, data_list[0], output)
        v_0 = torch.unsqueeze(torch.reshape(Fy_gradient, [-1]), 1).detach()

        # Hessian
        z_list = []
        output = out_f(data_list[1], params)
        Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], hparams, output, reg_f) 

        G_gradient = torch.reshape(params, [-1]) - args.eta*torch.reshape(Gy_gradient, [-1])
        # G_gradient = torch.reshape(params[0], [-1]) - args.eta*torch.reshape(Gy_gradient, [-1])
        
        for _ in range(args.hessian_q):
        # for _ in range(args.K):
            Jacobian = torch.matmul(G_gradient, v_0)
            v_new = torch.autograd.grad(Jacobian, params, retain_graph=True)[0]
            v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
            z_list.append(v_0)            
        v_Q = args.eta*v_0+torch.sum(torch.stack(z_list), dim=0)

        # Gyx_gradient
        output = out_f(data_list[2], params)
        Gy_gradient = gradient_gy(args, labels_list[2], params, data_list[2], hparams, output, reg_f)
        Gy_gradient = torch.reshape(Gy_gradient, [-1])
        Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), hparams, retain_graph=True)[0]
        outer_update = -Gyx_gradient 

        return outer_update

def gradient_fy(args, labels, params, data, output):
    loss = F.cross_entropy(output, labels)
    grad = torch.autograd.grad(loss, params)[0]
    return grad

def gradient_gy(args, labels_cp, params, data, hparams, output, reg_f):
    # loss = F.cross_entropy(output, labels_cp, reduction='none')
    loss = F.cross_entropy(output, labels_cp)
    loss_regu = reg_f(params, hparams, loss)
    grad = torch.autograd.grad(loss_regu, params, create_graph=True)[0]
    return grad
