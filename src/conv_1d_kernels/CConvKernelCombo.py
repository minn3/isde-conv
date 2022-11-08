import numpy as np

class CConvKernelCombo():

    def __init__(self, input_filters):
        self.input_filters = input_filters


    def comb_filter(self,x):
        xp = x.copy()
        for filter in self.input_filters:
            xp = filter.kernel(xp)
        return xp





