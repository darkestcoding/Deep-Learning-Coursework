import pandas as pd
import torch
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np

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


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.DoubleTensor)

    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    data = torch.tensor(df.iloc[:,[0,1,2,3]].values)
    data = data - data.mean(dim=0)
    U_1, V_1 = sgd_factorise_ad(data, rank=2)
    loss = torch.nn.functional.mse_loss(data, U_1 @ V_1.t(), reduction="sum")
    print("reconstruction loss: {}".format(loss))

    U,S,V = torch.svd(data)
    svd_data = U[:,:2] @ torch.diag(S[:2]) @ V[:,:2].t()
    # svd_data = U @ torch.diag(S) @ V.t()

    svd_loss = torch.nn.functional.mse_loss(data, svd_data, reduction="sum")
    print("svd loss: {}".format(svd_loss))

    print(svd_data.shape)
    plt.scatter(svd_data[:,0],svd_data[:,1])
    plt.title("PCA")
    plt.savefig("svd.png")
    plt.show()
    U_1 = U_1.detach().numpy()
    plt.scatter(U_1[:, 0], U_1[:, 1])
    plt.title("Factorised Matrix U")
    plt.savefig("U.png")
    plt.show()







