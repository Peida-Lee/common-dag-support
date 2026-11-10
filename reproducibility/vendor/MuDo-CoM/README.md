# codes

## Environment
Tested on Python 3.9.16.

```
pip install -r requirements.txt
```

## Numerical experiment
```
zn=10 #dimesion of latent variables
xn=1 #dimension of observations = xn*zn
T=2 #number of domains = T*zn
N=5000  #sample number per domain
k=2.0  #dependency level of latent, higher is stronger
noise='gauss'  #noise type in latent structure model


# linear latent structural model (default)
python main_numerical.py --n "$zn" --x-n "$xn" --T "$T" --N "$N" --graph-dense "$k" --noise-type "$noise"

# nonlinear latent structural model
python main_numerical.py --n "$zn" --x-n "$xn" --T "$T" --N "$N" --graph-dense "$k" --noise-type "$noise" --nonlinear
```

## fMRI experiment -- CamCAN
```
python main_camCAN.py
```
