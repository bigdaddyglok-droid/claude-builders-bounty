import numpy as np, torch
from uerf_brain import UERFField, SensoryProjection
from torchvision import datasets, transforms
to_t=transforms.ToTensor()
mn_tr=datasets.MNIST('./d',train=True,download=True); mn_te=datasets.MNIST('./d',train=False,download=True)
tr=[i for i,_ in mn_tr][:2000]; trl=torch.tensor([l for _,l in mn_tr][:2000])
mean_img=torch.stack([to_t(i).view(-1) for i in tr]).mean(0)
by={d:(trl==d).nonzero(as_tuple=True)[0].tolist() for d in range(10)}
te=[i for i,_ in mn_te][:300]; tel=[l for _,l in mn_te][:300]
def run(mode):
  torch.manual_seed(369)
  if mode=='precomputed': proj=SensoryProjection(784,64,seed=42,mean_raw=mean_img)
  else: proj=SensoryProjection(784,64,seed=42,online=True,ema=0.05)
  enc=lambda p: proj(to_t(p).view(-1))
  f=UERFField(n_max=200,n_initial=120,d=32,input_dim=64,n_classes=15); f.start_emergent()
  label={}
  for r in range(5):
    for d in range(10):
      x=enc(tr[by[d][r]])
      if d not in label:
        sig=(x.unsqueeze(-1)*f.c[:f.input_dim]).sum(0); label[d]=f.birth_category(sig)
      tv=torch.zeros(f.n_classes); tv[label[d]]=1.0
      f.experience(x,teaching_vector=tv,n_relax=6,domain_lo=0,domain_hi=f.n_classes)
  inv={v:d for d,v in label.items()}; lo,hi=min(label.values()),max(label.values())+1
  c=t=0
  for p,y in zip(te,tel):
    pr=f.eval_predict(proj(to_t(p).view(-1)),domain_lo=lo,domain_hi=hi)
    c+=int(inv.get(pr)==y); t+=1
  return 100*c/t
a=run('precomputed'); b=run('online')
print('precomputed-mean (offline): %.1f%%' % a)
print('online-mean (honest)      : %.1f%%' % b)
