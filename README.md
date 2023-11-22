# *UTILE-Oxy* - Deep Learning to Automate Video Analysis of Bubble Dynamics in Proton Exchange Membrane Electrolyzers

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/workflow.png)


We present  an automated workflow using deep learning for the analysis of videos containing oxygen bubbles in PEM electrolyzers by: 1. preparing annotated dataset and training models in order to conduct semantic segmentation of bubbles and 2. automating the extraction of bubble properties for further distribution analysis.

The pre-print [UTILE-Oxy - Deep Learning to Automate Video Analysis of Bubble Dynamics in Proton Exchange Membrane Electrolyzers](https://pubs.acs.org/) is available on ArXiv for further information!

## Description
This project focuses on the deep learning-based automatic analysis of polymer electrolyte membrane water electrolyzers (PEMWE) oxygen evolution videos. 
This repository contains the Python implementation of the UTILE-Oxy software for the automatic video analysis, feature-extraction and plotting.

The models that we present in this work are trained on a specific use-case scenario of interest in oxygen bubble evolution videos of transparent cells. It is possible to fine-tune, re-train or employ another model suitable for your individual case if your data has a strong visual deviation from the presented data here, which was recorded and shown as follows:

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/figexperiment.png)

## Model's benchmark
In our study, we trained several models to compare their prediction performance on unseen data. We trained specificalley three different models on the same dataset composed by :
- Standard U-Net 2D
- U-Net 2D with a ResNeXt 101 backbone 
- Attention U-Net

And we obtained the following performance results:

| Model                           | Precision [%] | Recall [%] | F1-Score [%] |
|---------------------------------|----------------|------------|--------------|
| U-Net 2D                        | 81             | 89         | 85           |
| U-Net with ResNeXt101 backbone  | 95             | 78         | 86           |
| Attention U-Net                 | 95             | 75         | 84           |


Since the F1-Scores are similar a visual inspection was carried out to find the best performing model :

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/benchmark.png)

But even clearer is the visual comparision of the running videos:

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/video_results.gif)

## Extracted features

### Time-resolved bubble ratio computation and bubble coverage distribution

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/timeresolved.png)

### Bubble position probability density map

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/heatmaps.png)

### Individual bubble shape analysis

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/individualcorrect.png)

## Installation
In order to run the actual version of the code, the following steps need to be done:
- Clone the repository
- Create a new environment using Anaconda using Python 3.8 or superior
- Pip install the jupyter notebook library

    ```
    pip install notebook
    ```
- From your Anaconda console open jupyter notebook (just tip "jupyter notebook" and a window will pop up)
- Open the /UTILE-Oxy/UTILE-Oxy_prediction.ipynb file from the jupyter notebook directory
- Further instructions on how to use the tool are attached to the code with examples in the juypter notebook

## Dependencies
The following libraries are needed to run the program:

  ```
   pip install opencv-python, numpy, patchify, pillow, segmentation_models, keras, tensorflow, matplotlib, scikit-learn, pandas

   ```
### Notes
