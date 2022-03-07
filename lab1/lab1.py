from typing import Tuple
import torch

def sgd_factorise(A:torch.Tensor, rank:int, num_epochs=1000, lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m,n = A.shape
    U = torch.rand(m, rank)
    V = torch.rand(n,rank)
    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                e = A[r, c] - U[r] @ V[c].t()
                U[r] = U[r] + lr * e * V[c]
                V[c] = V[c] + lr * e * U[r]
    return U, V

def mse_loss(A,U,V):
    mse = torch.nn.MSELoss(reduction="sum")
    loss = mse(U @ V.t(), A)
    return loss

if __name__ == '__main__':
    A = torch.tensor([[0.3374, 0.6005, 0.1735], [3.3359, 0.0492, 1.8374], [2.9407, 0.5301, 2.2620]])
    U,V = sgd_factorise(A=A,rank = 2)
    print(mse_loss(A,U,V))
    print(A)
    # print(U@V.t())
    # print(U,V)
    U_1,S_1,V_1 = torch.svd(A)
    S_1[2] = 0
    mse = torch.nn.MSELoss(reduction="sum")
    mse(U_1 @ torch.diag(S_1) @ V_1.t(), A)
    print(mse(U_1@torch.diag(S_1)@V_1.t(),A))
    # print(U_1@torch.diag(S_1)@V_1.t())







