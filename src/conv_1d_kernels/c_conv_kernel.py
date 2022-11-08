import numpy as np
from abc import ABC, abstractmethod


class CConvKernel(ABC):

    def __init__(self, kernel_size):

        self._mask = None
        self.kernel_size = kernel_size

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size):
        if (kernel_size < 2):
            raise ValueError("Error the value must be greter than 2 ")
        elif (kernel_size % 2 == 0):
            raise ValueError("Error the value must be an odd number ")
        self._kernel_size = kernel_size
        self.kernel_mask()
    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self,mask):
        self._mask = mask

    @abstractmethod
    def kernel_mask(self):
        raise NotImplementedError("The method is abstract")


    def kernel(self,x):

        xp = x.copy()
        offset = int((self.kernel_size-1)/2)
        for i in range(offset,(len(x)-offset)):
            xp[i] = np.dot(self.mask,x[i-offset:i+offset+1]).sum()

        return xp
