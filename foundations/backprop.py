import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(x,w)+b
        z = 1/(1+np.exp(-z))
        dldb = (z-y_true)*z*(1-z)
        dldw = dldb*x
        dldb = np.round(dldb,5)
        dldw = np.round(dldw,5)
        return (dldw,dldb)

