import torch





a = torch.linspace(-4,4,1000)
uu = []
for i in range(1000):
    uu.append(torch.cat((a[i]*torch.ones_like(b).unsqueeze(-1), b.unsqueeze(-1)),dim=-1))
u = torch.cat(uu)
print(u.shape)
print(u[:10])