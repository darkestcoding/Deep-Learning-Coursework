from typing import Tuple
import torch

def sgd_factorise_ad(A:torch.Tensor, rank:int, num_epochs=1000, lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = A.shape
    U = torch.rand(m, rank, requires_grad=True)
    V = torch.rand(n, rank, requires_grad=True)
    for epoch in range(num_epochs):
        U.grad = V.grad = None
        loss = torch.nn.functional.mse_loss(A, U @ V.t(), reduction="sum")
        loss.backward()
        U.data = U - lr * U.grad
        V.data = V - lr * V.grad
    return U, V

def mse_loss(A,U,V):
    mse = torch.nn.MSELoss(reduction="sum")
    loss = mse(U @ V.t(), A)
    return loss

if __name__ == '__main__':
    A = torch.tensor([[0.3374, 0.6005, 0.1735], [3.3359, 0.0492, 1.8374], [2.9407, 0.5301, 2.2620]])
    U,V = sgd_factorise_ad(A=A, rank = 2)

    print(U @ V.t())


