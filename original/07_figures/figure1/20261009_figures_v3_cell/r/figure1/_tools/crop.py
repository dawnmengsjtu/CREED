# usage: crop.py in.png out.png x0_mm y0_mm x1_mm y1_mm [scale]
import sys
from PIL import Image
im=Image.open(sys.argv[1]);mm=600/25.4
x0,y0,x1,y1=[int(float(v)*mm) for v in sys.argv[3:7]]
c=im.crop((x0,y0,x1,y1))
s=float(sys.argv[7]) if len(sys.argv)>7 else 1.0
if s!=1.0:c=c.resize((int(c.width*s),int(c.height*s)),Image.LANCZOS)
c.save(sys.argv[2])
