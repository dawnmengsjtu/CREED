# usage: extent.py page.png x0_mm x1_mm [y0_mm y1_mm]  -> prints ink runs (mm) along y within the x range
import sys
from PIL import Image
import numpy as np
im=np.asarray(Image.open(sys.argv[1]).convert('L'))
H,W=im.shape; mm=600/25.4
x0,x1=[int(float(v)*mm) for v in sys.argv[2:4]]
y0=int(float(sys.argv[4])*mm) if len(sys.argv)>4 else 0
y1=int(float(sys.argv[5])*mm) if len(sys.argv)>5 else H
sub=im[y0:y1,x0:x1]<235
rows=sub.any(axis=1)
runs=[];start=None
for i,r in enumerate(rows):
    if r and start is None:start=i
    if not r and start is not None:runs.append((start,i));start=None
if start is not None:runs.append((start,len(rows)))
print('page px',W,H,'mm',W/mm,H/mm)
for a,b in runs:
    if (b-a)>0: print('ink %.2f - %.2f mm'%((a+y0)/mm,(b+y0)/mm))
