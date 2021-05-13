# NetStruct Visualization
This repository contains some helper code for data visualization in NetStruct applications onto cultural variation datasets.
_____

NetStruct Hierarchy is a network-based hierarchical clustering tool developed for population structure analysis, proposed by Greenbaum *et al.* in paper [Network-based hierarchical population structure analysis for large genomic datasets](https://genome.cshlp.org/content/early/2019/11/05/gr.250092.119). The softward is available at https://github.com/amirubin87/NetStruct_Hierarchy.

The usage of this framework is generalized onto cultural variation datasets, including English pronunciation differences, folklore mythology variation, phoneme inventory, and frequency data of baby name at birth. 

The code snippets in `netstruct_vis.py` help visualize the clustering results with patterns of variation in the datasets. 
Visualization formats provided include colored tree, annotated map, and tree with pie-chart nodes.

Two demos are provided. Data in output files under `data` contain the clustering results generated by NetStruct, as well as meta information of the individuals clustered in these two demos. 
_____
### Demo 1: [Clustering of cultural groups based on folklore and mythology motifs](VisDemo1.ipynb)

_____
### Demo 2: [Clustering of popular female names based on frequency data from 1880 to 2019](VisDemo2.ipynb)


