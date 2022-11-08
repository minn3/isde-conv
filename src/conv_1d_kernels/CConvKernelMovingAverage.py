import numpy as np
from src.conv_1d_kernels.c_conv_kernel import CConvKernel

class CConvKernelMovingAverage(CConvKernel):

    def __init__(self, kernel_size):
        super().__init__(kernel_size)


    def kernel_mask(self):
        self.mask = np.ones(self.kernel_size) / self.kernel_size
