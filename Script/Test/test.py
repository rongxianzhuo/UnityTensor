import torch
import struct
import torch.nn as nn
import torch.optim as optim


class TensorWriter(object):

    def __init__(self, file_path):
        self.file = open(file_path, "wb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def write(self, *tensor_list):
        for tensor in tensor_list:
            for f in tensor.reshape(-1).detach().numpy():
                self.file.write(struct.pack('f', float(f)))

    def close(self):
        self.file.close()


def save_tensor_list(file_name, *tensor_list):
    with open(file_name, 'wb') as file:
        for tensor in tensor_list:
            for f in tensor.reshape(-1).detach().numpy():
                file.write(struct.pack('f', float(f)))


def linspace_tensor(requires_grad, *shape, ):
    s = 1
    for i in shape:
        s *= i
    t = torch.linspace(1.0, float(s), s).reshape(shape)
    t.requires_grad = requires_grad
    return t


def add():
    a = torch.linspace(1.0, 2.0, 2).reshape([1, 2, 1])
    ab = a.broadcast_to([1, 2, 4])
    at = ab.transpose(1, 2)
    b = torch.linspace(10.0, 80.0, 8).reshape([2, 1, 4, 1, 1])
    bt = b.transpose(2, 3)
    bb = bt.broadcast_to([2, 3, 1, 4, 1])
    c = at + bb
    ct = c.transpose(1, 3)
    d = torch.linspace(100.0, 2000.0, 12).reshape([4, 1, 3, 1])
    e = ct + d
    save_tensor_list(".\\..\\..\\Resources\\UT\\Test\\TestAdd.bytes", a, ab, at, b, bt, bb, c, ct, d, e)


def sum():
    with TensorWriter(".\\..\\..\\Resources\\UT\\Test\\Sum.bytes") as writer:
        a = linspace_tensor(True, 2, 3, 4, 5)
        b = torch.sum(a, (1, 3), keepdim=True)
        t = linspace_tensor(False, 2, 1, 4, 1)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(b, t)
        loss.backward()
        writer.write(a, t, b, a.grad.data)


def mse_loss():
    a = torch.randn([2, 4, 1], requires_grad=True)
    at = a.transpose(1, 2)
    b = torch.randn([2, 5, 4], requires_grad=True)
    c = at + b
    t = torch.randn([2, 5, 4])
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(c, t)
    loss.backward()
    save_tensor_list(".\\..\\..\\Resources\\UT\\Test\\MseLoss.bytes", a, b, t, a.grad.data, b.grad.data)


def adam():
    with TensorWriter(".\\..\\..\\Resources\\UT\\Test\\Adam.bytes") as writer:
        a = linspace_tensor(True, 16, 3, 2)
        at = a.transpose(1, 2)
        b = linspace_tensor(True, 2, 3)
        c = at + b
        t = linspace_tensor(False, 16, 2, 3)
        writer.write(a, b, t)
        loss_fn = torch.nn.MSELoss()
        optimizer = optim.Adam([a, b], lr=1.0, eps=0.001, betas=(0.8, 0.9))
        loss = loss_fn(c, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.write(a)


def matmul():
    with TensorWriter(".\\..\\..\\Resources\\UT\\Test\\MatMul.bytes") as writer:
        a = torch.randn((2, 3, 4, 5), requires_grad=True)
        b = torch.randn((2, 3, 5, 6), requires_grad=True)
        c = torch.matmul(a, b)
        t = torch.randn((2, 3, 4, 6), requires_grad=True)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(c, t)
        loss.backward()
        writer.write(a, b, t, c, a.grad.data)


def mul():
    with TensorWriter(".\\..\\..\\Resources\\UT\\Test\\Mul.bytes") as writer:
        a = linspace_tensor(True, 16, 10)
        b = linspace_tensor(True, 10)
        c = a * b
        t = linspace_tensor(False, 16, 10)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(c, t)
        loss.backward()
        writer.write(a, b, t, c, a.grad.data)


def linear():
    with TensorWriter(".\\..\\..\\Resources\\UT\\Test\\Linear.bytes") as writer:
        a = linspace_tensor(True, 3, 5, 8)
        l = nn.Linear(8, 6)
        b = l(a)
        c = nn.ReLU()(b)
        t = linspace_tensor(False, 3, 5, 6)
        loss_fn = torch.nn.MSELoss()
        optimizer = optim.Adam([l.weight, l.bias], lr=1.0, eps=0.001, betas=(0.8, 0.9))
        loss = loss_fn(c, t)
        optimizer.zero_grad()
        loss.backward()
        writer.write(a, l.weight.data, l.bias.data, t, b, c, l.bias.grad.data)
        optimizer.step()
        writer.write(l.bias.data, l.weight.data)


def max():
    with TensorWriter(".\\..\\..\\Resources\\UT\\Test\\Max.bytes") as writer:
        a = torch.randn((2, 3, 4, 5), requires_grad=True)
        b, c = torch.max(a, dim=1, keepdim=True)
        t = torch.randn((2, 1, 4, 5))
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(b, t)
        loss.backward()
        writer.write(a, t, b, a.grad.data)


def contiguous():
    with TensorWriter(".\\..\\..\\Resources\\UT\\Test\\Contiguous.bytes") as writer:
        a = torch.randn((2, 1, 5), requires_grad=True)
        b = a.broadcast_to((2, 3, 5)).transpose(1, 2)
        c = b.contiguous()
        t = torch.randn((2, 5, 3))
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(c, t)
        loss.backward()
        writer.write(a, t, c, a.grad.data)



if __name__ == '__main__':
    matmul()
