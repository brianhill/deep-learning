# README

*envs and resourcces used by the project*

## The raschka env

It might be best to ditch conda altogether and start from scratch with the 
recommendations at [https://pytorch.org](https://pytorch.org]), but I did
the following:

```
conda create -n raschka python=3.12.7 --verbose
conda activate raschka
pip install torch
pip install numpy
```

The log showed that torch 2.7.1 was installed. Raschka used 2.4.0 when writing his book, and
he recommends using that exact version.

It was surprising that numpy was not installed as a dependency of torch. Installing it got
version 2.3.1.

## Resources for Raschka

```
git clone git@github.com:rasbt/llms-from-scratch
```

## The keras env

```
conda create -n keras python=3.11.5 --verbose
conda activate keras
pip install keras
conda install tensorflow==2.18.1
conda list | grep -E "keras|tensorflow|numpy"
```

```
keras                     3.10.0                   pypi_0    pypi
numpy                     2.3.1                    pypi_0    pypi
numpy-base                2.0.1           py311hfbfe69c_1  
tensorflow                2.18.1          cpu_py311h18e2dce_0  
tensorflow-base           2.18.1          cpu_py311h97f355d_0  
```
