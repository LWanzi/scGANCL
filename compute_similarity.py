import torch
from scipy import spatial
import numpy as np
a=torch.randn(2,2)
print(a)
b=torch.randn(3,2)
print(b)
a_norm=a/a.norm(dim=1)[:,None]
b_norm=b/b.norm(dim=1)[:,None]
res=torch.mm(a_norm,b_norm.transpose(0,1))
print(res)