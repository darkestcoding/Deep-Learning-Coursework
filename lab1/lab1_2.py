from typing import Tuple
import torch

def sgd_factorise_masked(A:torch.Tensor, M:torch.Tensor, rank:int, num_epochs=1000, lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = A.shape
    U = torch.rand(m, rank)
    V = torch.rand(n, rank)
    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                if M[r, c] != 0:
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
    M = torch.tensor([[1,1,1],[0,1,1],[1,0,1]])
    U,V = sgd_factorise_masked(A=A, M=M, rank = 2)
    print(U @ V.t())
    print(A)
    print(mse_loss(A,U,V))








