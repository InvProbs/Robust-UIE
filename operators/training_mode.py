""" code borrowed from SGD jittering project """

import torch.nn as nn
import torch


def norms(Z, dim=1):
    if dim == 1:
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None]
    elif dim == 2:
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None]
    elif dim == 3:
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]
    else:
        return None


def norms_2D(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None]


def norms_3D(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


def PGD(net, X, y, epsilon, alpha, num_iter, dim=1, eps_mode='l2', nouts=3):
    """modified from https://adversarial-ml-tutorial.org/adversarial_examples/"""
    net.eval()
    delta = torch.zeros_like(y, requires_grad=True)
    loss_list = []
    if eps_mode == 'l2':
        for t in range(num_iter):
            if nouts == 1:
                Xk = net(y + delta)
            elif nouts == 2:
                Xk, _ = net(y + delta)
            elif nouts == 3:
                Xk, _, _ = net(y + delta)
            loss = nn.MSELoss()(Xk, X)
            loss_list.append(loss.item())
            loss.backward()
            delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach(), dim=dim)
            # delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
            delta.data *= epsilon / norms(delta.detach(), dim=dim).clamp(min=epsilon)
            delta.grad.zero_()
            # delta = delta.detach()
            # delta.require_grad_(True)
    elif eps_mode == 'inf':
        for t in range(num_iter):
            if nouts == 1:
                Xk = net(y + delta)
            elif nouts == 2:
                Xk, _ = net(y + delta)
            elif nouts == 3:
                Xk, _, _ = net(y + delta)
            loss = nn.MSELoss()(Xk, X)
            loss_list.append(loss.item())
            loss.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
    return delta.detach(), loss_list


def PGD_v2(net, X, forward_op, y, epsilon, alpha, num_iter, dim=2, eps_mode='l2'):
    """https://adversarial-ml-tutorial.org/adversarial_examples/"""
    delta = torch.zeros_like(y, requires_grad=True)
    loss_list = []
    if eps_mode == 'l2':
        for t in range(num_iter):
            loss = nn.MSELoss()(net(forward_op.adjoint(y + delta)), X)
            loss_list.append(loss.item())
            loss.backward()
            delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach(), dim=dim)
            # delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
            delta.data *= epsilon / norms(delta.detach(), dim=dim).clamp(min=epsilon)
            delta.grad.zero_()
    elif eps_mode == 'inf':
        for t in range(num_iter):
            loss = nn.MSELoss()(net(forward_op.adjoint(y + delta)), X)
            loss.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
    return delta.detach(), loss_list

def PGD_v3(net, y, X, epsilon, alpha, num_iter, dim=3):
    """https://adversarial-ml-tutorial.org/adversarial_examples/"""
    delta = torch.zeros_like(y, requires_grad=True)
    loss_list = []
    for t in range(num_iter):
        loss = nn.MSELoss()(net.sample(y+delta, y+delta), X)
        loss_list.append(loss.item())
        loss.backward()
        delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach(), dim=dim)
        # delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach(), dim=dim).clamp(min=epsilon)
        delta.grad.zero_()
    return delta.detach(), loss_list


def PGD_adaptLU(net, X, y, epsilon, alpha, num_iter, dim=1, eps_mode='l2'):
    """modified from https://adversarial-ml-tutorial.org/adversarial_examples/"""
    net.eval()
    delta = torch.zeros_like(y, requires_grad=True)
    loss_list = []
    if eps_mode == 'l2':
        for t in range(num_iter):
            net_input = y + delta
            Xk = net(net_input, net_input, net_input)
            loss = nn.MSELoss()(Xk, X)
            loss_list.append(loss.item())
            loss.backward()
            delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach(), dim=dim)
            # delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
            delta.data *= epsilon / norms(delta.detach(), dim=dim).clamp(min=epsilon)
            delta.grad.zero_()
            # delta = delta.detach()
            # delta.require_grad_(True)
    elif eps_mode == 'inf':
        for t in range(num_iter):
            net_input = y + delta
            Xk = net(net_input, net_input, net_input)
            loss = nn.MSELoss()(Xk, X)
            loss_list.append(loss.item())
            loss.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
    return delta.detach(), loss_list


# def fgsm_attack(image, epsilon, data_grad, dim=3):
#     """ Code modified from: https://pytorch.org/tutorials/beginner/fgsm_tutorial.html """
#     # sign_data_grad = data_grad.sign()
#     scale = norms(data_grad.detach(), dim=dim)
#     perturbed_image = image + epsilon / scale * data_grad
#     # perturbed_image = image + epsilon * sign_data_grad
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     return perturbed_image

def fgsm_attack(net, y, X, epsilon, alpha, num_iter, criteria, dim=3):
    delta = torch.zeros_like(y)
    loss_list = []
    for t in range(num_iter):
        net_input = (y + delta).detach()
        Xk, _, _ = net(net_input)
        # Xk = invNet(net_input, net_input, net_input)
        data_grad = net_input.grad.data.detach()
        net_input = net_input + alpha * data_grad
        delta = net_input - y if norms(net_input - y, dim=dim) <= epsilon else epsilon * (net_input - y) / norms(net_input - y, dim=dim)
        # delta = epsilon * (net_input - y) / norms(net_input - y, dim=dim)

        net_input = (y + delta).detach()
        Xk, _, _ = net(net_input).detach()
        # Xk = invNet(net_input, net_input, net_input).detach()
        loss_list.append(criteria(Xk, X).item())
    return delta.detach(), loss_list

def fgsm_attack_adaptLU(invNet, y, X, epsilon, alpha, num_iter, criteria, dim=3):
    delta = torch.zeros_like(y)
    loss_list = []
    for t in range(num_iter):
        net_input = (y + delta).detach()
        # Xk = invNet(net_input)
        Xk = invNet(net_input, net_input, net_input)
        data_grad = net_input.grad.data.detach()
        net_input = net_input + alpha * data_grad
        delta = net_input - y if norms(net_input - y, dim=dim) <= epsilon else epsilon * (net_input - y) / norms(net_input - y, dim=dim)
        # delta = epsilon * (net_input - y) / norms(net_input - y, dim=dim)

        net_input = (y + delta).detach()
        # Xk = invNet(net_input).detach()
        Xk = invNet(net_input, net_input, net_input).detach()
        loss_list.append(criteria(Xk, X).item())
    return delta.detach(), loss_list
# plt.plot(loss_list)


def PGD_Ucolor(net, X, y, D, epsilon, alpha, num_iter, dim=1):
    """modified from https://adversarial-ml-tutorial.org/adversarial_examples/"""
    net.eval()
    delta = torch.zeros_like(y, requires_grad=True)
    loss_list = []
    for t in range(num_iter):
        Xk = net(y + delta, D)
        loss = nn.MSELoss()(Xk, X)
        loss_list.append(loss.item())
        loss.backward()
        delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach(), dim=dim)
        # delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach(), dim=dim).clamp(min=epsilon)
        delta.grad.zero_()
    return delta.detach(), loss_list