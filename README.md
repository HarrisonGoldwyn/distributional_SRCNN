# Distributional Super-Resolution Convolutional Neural Network

This repo contains the implementation of a CNN trained on a multidimentional distributional loss function, will code and data to reproduce all results, analysis, and figures contained in submitted paper [**Multidimensional Distributional Neural Network Output Demonstrated in Super-Resolution of Surface Wind Speed**](arxiv.org/our_paper)

## Repo Structure
This repository contains our 3-stage framework for estimating mean and covariance with a CNN trained on a multidimensional Gaussian loss function over heteroscedastic data. Analysis for the publication was conducted in directory [sr_ordered_train0p75/](./sr_ordered_train0p75), named after the "subregion-ordered" data with a 75/25 train/test split. The specific data used is discussed in the publication. 

Within [sr_ordered_train0p75/](./sr_ordered_train0p75), each stage of the framework is contained within its own directory. To reproduce results, python files in each stage must be run sequentially. Jupyter notebooks for analysis post-training are contained in the [analysis/](.sr_ordered_train0p75/analysis) and [sample_generation/](.sr_ordered_train0p75/sample_generation) directories.

## Data
As a test case for our method, we studied a subset of the Surface wind speed data extracted from high-resolution, convection-permitting model simulations produced as part of the UK Natural Environment Research Council (NERC) Cascade project [NERC 2008].  
We use the 4~km resolution Cascade simulation over the tropical Indo-Pacific Warm Pool. 

The subset of data used for our analysis is contained in [data/](./data), shared under consistency with the public data license associated with [CASCADE data](https://catalogue.ceda.ac.uk/uuid/20981e3052a66ca71c2ba92b94760150/).

## Dependencies
- dl-kit (HJG/dev)
- pytorch

## References
Natural Environment Research Council (NERC); NERC CASCADE Project participants; Lister, G.; Woolnough, S. (2008): Cascade - Scale interactions in the tropical atmosphere model runs. NCAS British Atmospheric Data Centre, date of citation.