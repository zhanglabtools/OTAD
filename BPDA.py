"""Backward Pass Differentiable Approximation (BPDA) and PGD attack."""
from abc import ABCMeta
import numpy as np
import torch
from torch import autograd
import torch.nn as nn

def _wrap_forward_as_function_forward(forward):
    def function_forward(ctx, x):
        ctx.save_for_backward(x)
        return forward(x)
    return staticmethod(function_forward)


def _wrap_backward_as_function_backward(backward):
    def function_backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return backward(grad_output, x)
    return staticmethod(function_backward)


def _create_identity_function():
    def identity(grad_output, x):
        return grad_output
    return identity


def _create_backward_from_forwardsub(forwardsub):
    def backward(grad_output, x):
        x = x.detach().clone().requires_grad_()
        grad_output = grad_output.detach().clone()
        with torch.enable_grad():
            y = forwardsub(x)
            return autograd.grad(y, x, grad_output)[0].detach().clone()
    return backward

class BPDAWrapper(nn.Module):
    """Wrap forward module with BPDA backward path."""

    def __init__(self, forward, forwardsub=None, backward=None):
        super(BPDAWrapper, self).__init__()

        if forwardsub is not None:
            backward = _create_backward_from_forwardsub(forwardsub)
        else:
            if backward is None:
                backward = _create_identity_function()

        self._create_autograd_function_class()
        self._Function.forward = _wrap_forward_as_function_forward(forward)
        self._Function.backward = _wrap_backward_as_function_backward(backward)

    def forward(self, *args, **kwargs):
        return self._Function.apply(*args, **kwargs)

    def _create_autograd_function_class(self):
        class _Function(autograd.Function):
            pass
        self._Function = _Function
        
        
class Attack(object):
    """Abstract base class for all attack classes."""

    __metaclass__ = ABCMeta

    def __init__(self, predict, loss_fn, clip_min, clip_max):
        self.predict = predict
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, **kwargs):
        """Generate adversarial examples. Override in subclasses."""
        raise NotImplementedError("Sub-classes must implement perturb.")

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)

def replicate_input(x):
    return x.detach().clone()
def is_float_or_torch_tensor(x):
    return isinstance(x, torch.Tensor) or isinstance(x, float)
class LabelMixin(object):
    def _get_predicted_label(self, x):
        """Compute predicted labels to prevent label leaking."""
        with torch.no_grad():
            outputs = self.predict(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def _verify_and_process_inputs(self, x, y):
        if self.targeted:
            assert y is not None

        if not self.targeted:
            if y is None:
                y = self._get_predicted_label(x)

        x = replicate_input(x)
        y = replicate_input(y)
        return x, y        
    
def clamp(input, min=None, max=None):
    if min is not None and max is not None:
        return torch.clamp(input, min=min, max=max)
    elif min is None and max is None:
        return input
    elif min is None and max is not None:
        return torch.clamp(input, max=max)
    elif min is not None and max is None:
        return torch.clamp(input, min=min)
    else:
        raise ValueError("This is impossible") 
        
def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Multiply each sample in batch_tensor by corresponding scalar in vector."""
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor

def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)

def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)

def _batch_clamp_tensor_by_vector(vector, batch_tensor):
    """Clamp each sample in batch_tensor by corresponding value in vector."""
    return torch.min(
        torch.max(batch_tensor.transpose(0, -1), -vector), vector
    ).transpose(0, -1).contiguous()
        
def rand_init_delta(delta, x, ord, eps, clip_min, clip_max):
    """Random initialization for perturbation delta."""
    if isinstance(eps, torch.Tensor):
        assert len(eps) == len(delta)

    if ord == np.inf:
        delta.data.uniform_(-1, 1)
        delta.data = batch_multiply(eps, delta.data)
    elif ord == 2:
        delta.data.uniform_(0, 1)
        delta.data = delta.data - x
        delta.data = clamp_by_pnorm(delta.data, ord, eps)
    else:
        error = "Only ord = inf and ord = 2 have been implemented"
        raise NotImplementedError(error)

    delta.data = clamp(
        x + delta.data, min=clip_min, max=clip_max) - x
    return delta.data

def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor
def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """Normalize gradients by p-norm."""
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)
def perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0):
    """Iterative PGD perturbation (supports Linf and L2)."""
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(xvar)

    delta.requires_grad_()
    for ii in range(nb_iter):
        outputs = predict(xvar + delta)
        loss = loss_fn(outputs, yvar)
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data

        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                               ) - xvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)
        else:
            error = "Only ord = inf and ord = 2 have been implemented"
            raise NotImplementedError(error)

        delta.grad.data.zero_()

    x_adv = clamp(xvar + delta, clip_min, clip_max)
    return x_adv

class PGDAttack(Attack, LabelMixin):
    """PGD attack (Madry et al, 2017). Supports Linf and L2."""

    def __init__(
            self, predict, loss_fn=None, eps=0.3, nb_iter=40,
            eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, targeted=False):
        super(PGDAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

    def perturb(self, x, y=None):
        """Generate adversarial examples with perturbation budget eps."""
        x, y = self._verify_and_process_inputs(x, y)

        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta)
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x

        rval = perturb_iterative(
            x, y, self.predict, nb_iter=self.nb_iter,
            eps=self.eps, eps_iter=self.eps_iter,
            loss_fn=self.loss_fn, minimize=self.targeted,
            ord=self.ord, clip_min=self.clip_min,
            clip_max=self.clip_max, delta_init=delta)

        return rval.data    
    