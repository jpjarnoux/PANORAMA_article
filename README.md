# Panorama: A robust pangenome-based method for predicting and comparing biological systems across species

## About the project

PANORAMA is an innovative pangenomic tool designed to exploit pangenome graphs and enable them to be annotated and
compared in order to explore the genomic diversity of several species. Based on the **PPanGGOLiN software suite**¹
([https://github.com/labgem/PPanGGOLiN](https://github.com/labgem/PPanGGOLiN)), PANORAMA integrates advanced methods for
rule-based prediction of macromolecular systems and comparative analysis of conserved features between different
pangenomes, such as hotspots of insertion. We illustrate the use of PANORAMA on a *Pseudomonas aeruginosa* dataset,
evaluating its performance against reference defense system prediction tools such as PADLOC² and DefenseFinder³.
The analysis was then extended to a larger set including four species of Enterobacteriaceae (>6,000 genomes),
demonstrating PANORAMA's ability to annotate, compare and explore the diversity and distribution of biological systems
across multiple species. This work provides new methods for the large-scale comparative study of microbial genomes and
underlines the relevance of pangenome approaches in deciphering their evolutionary dynamics.
PANORAMA is freely available and accessible through:
[https://github.com/labgem/PANORAMA](https://github.com/labgem/PANORAMA)

## Authors

- Jérôme Arnoux, LABGeM, Genomics Metabolics, CEA, Genoscope, Institut François Jacob, Université d’Évry, Université
  Paris-Saclay, CNRS
- Jean Mainguy, LABGeM, Genomics Metabolics, CEA, Genoscope, Institut François Jacob, Université d’Évry, Université
  Paris-Saclay, CNRS
- Laura Bry, LABGeM, Genomics Metabolics, CEA, Genoscope, Institut François Jacob, Université d’Évry, Université
  Paris-Saclay, CNRS
- Quentin Fernandez De Grado, LABGeM, Genomics Metabolics, CEA, Genoscope, Institut François Jacob, Université d’Évry,
  Université Paris-Saclay, CNRS
- David Vallenet, LABGeM, Genomics Metabolics, CEA, Genoscope, Institut François Jacob, Université d’Évry, Université
  Paris-Saclay, CNRS
- Alexandra Calteau, LABGeM, Genomics Metabolics, CEA, Genoscope, Institut François Jacob, Université d’Évry, Université
  Paris-Saclay, CNRS

## Dependencies

All the dependencies has been listed in the conda-env.yml file.

## Dataset

<!-- This part must be done when the repository with the data will be created-->

## Running the project

To begin, note that you need to install Conda, follow the installation
guide [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

To execute the notebook, you will need to first install some packages.

These are listed in the following conda environment file conda-env.yml, that you could use to easily install them.

You can follow the under code to install all dependencies and run the notebook:

```shell
git clone https://github.com/labgem/PANORAMA_article
cd PANORAMA_article
conda update -n base -c defaults conda -y
conda env create --file conda-env.yml
conda init bash
conda activate panorama_notebook
python -m ipykernel install --user --name=panorama_notebook
jupyter notebook --notebook-dir=.
```

*N.B: If you encounter any difficulty to install with conda try to use mamba or write an issue*

## References

1. Gautreau, Guillaume, Adelme Bazin, Mathieu Gachet, Rémi Planel, Laura Burlot, Mathieu Dubois, Amandine Perrin, *et
   al.* « PPanGGOLiN: Depicting Microbial Diversity via a Partitioned Pangenome Graph ». PLOS Computational Biology 16,
   nᵒ 3 (19 mars 2020): e1007732. https://doi.org/10.1371/journal.pcbi.1007732.
2. Payne, Leighton J, Thomas C Todeschini, Yi Wu, Benjamin J Perry, Clive W Ronson, Peter C Fineran, Franklin L Nobrega,
   et Simon A Jackson. « Identification and classification of antiviral defence systems in bacteria and archaea with
   PADLOC reveals new system types ». Nucleic Acids Research 49, nᵒ 19 (8 novembre 2021):
   10868‑78. https://doi.org/10.1093/nar/gkab883.
3. Doron, Shany, Sarah Melamed, Gal Ofir, Azita Leavitt, Anna Lopatina, Mai Keren, Gil Amitai, et Rotem Sorek. «
   Systematic discovery of antiphage defense systems in the microbial pangenome ». Science 359, nᵒ 6379 (2 mars 2018):
   eaar4120. https://doi.org/10.1126/science.aar4120.
