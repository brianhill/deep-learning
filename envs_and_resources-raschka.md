# Envs and Resources for Raschka

*Document the dependencies of the projects*

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
