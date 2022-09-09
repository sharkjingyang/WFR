from math import *
import torch

n = 100000
a = torch.randn(n)
b = torch.randn(n)
med2 = torch.sum((a-b)**2)/n
print(med2)

d= sqrt(4/2/log(4096))
print(d)