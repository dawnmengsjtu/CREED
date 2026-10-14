"""Multi-output task kernels with missing labels retained, grouped exact masks."""
import numpy as np
from scipy.linalg import solve


def predict_masked(kernel,cross,y,alpha,min_observed=20):
    y=np.asarray(y,float)
    if y.ndim!=2 or kernel.shape!=(len(y),len(y)) or cross.shape[1]!=len(y):raise ValueError('kernel/target shape mismatch')
    out=np.full((len(cross),y.shape[1]),np.nan);groups={}
    for j in range(y.shape[1]):
        mask=np.isfinite(y[:,j])
        if mask.sum()<min_observed:continue
        groups.setdefault(mask.tobytes(),[]).append(j)
    for maskbytes,cols in groups.items():
        mask=np.frombuffer(maskbytes,dtype=bool);indices=np.flatnonzero(mask)
        yy=y[np.ix_(indices,cols)];mean=yy.mean(0)
        k=kernel[np.ix_(indices,indices)].copy();k.flat[::len(k)+1]+=alpha
        coef=solve(k,yy-mean,assume_a='pos')
        out[:,cols]=cross[:,indices]@coef+mean
    return out
