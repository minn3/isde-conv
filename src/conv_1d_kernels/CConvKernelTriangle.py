import numpy as np
from src.conv_1d_kernels.c_conv_kernel import CConvKernel

class CConvKernelTriangle(CConvKernel):

    def __init__(self, kernel_size):
        super().__init__(kernel_size)


    def kernel_mask(self):
        offset = int((self.kernel_size-1)/2)
        self.mask = np.ones(self.kernel_size)
        for i in range(1,offset+1):
            self.mask[i] = self.mask[i-1] +1

        for i in range(offset+1, len(self.mask)):
                self.mask[i] = self.mask[i - 1] -1

        self.mask =  1/self.mask.sum()
