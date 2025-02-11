# *UTILE-Oxy* - Deep Learning to Automate Video Analysis of Bubble Dynamics in Proton Exchange Membrane Electrolyzers

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/workflow.png)


We present  an automated workflow using deep learning for the analysis of videos containing oxygen bubbles in PEM electrolyzers by 1. preparing an annotated dataset and training models in order to conduct semantic segmentation of bubbles and 2. automating the extraction of bubble properties for further distribution analysis.

The publication [UTILE-Oxy - Deep Learning to Automate Video Analysis of Bubble Dynamics in Proton Exchange Membrane Electrolyzers](https://pubs.rsc.org/en/content/articlelanding/2024/cp/d3cp05869g) is available in an open access fashion on the journal PCCP for further information!


## Description
This project focuses on the deep learning-based automatic analysis of polymer electrolyte membrane water electrolyzers (PEMWE) oxygen evolution videos. 
This repository contains the Python implementation of the UTILE-Oxy software for automatic video analysis, feature extraction, and plotting.

The models we present in this work are trained on a specific use-case scenario of interest in oxygen bubble evolution videos of transparent cells. It is possible to fine-tune, re-train or employ another model suitable for your individual case if your data has a strong visual deviation from the presented data here, which was recorded and shown as follows:

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/figexperiment.png)

## Model's benchmark
In our study, we trained several models to compare their prediction performance on unseen data. We trained specifically three different models on the same dataset composed by :
- Standard U-Net 2D
- U-Net 2D with a ResNeXt 101 backbone 
- Attention U-Net

We obtained the following performance results:

| Model                           | Precision [%] | Recall [%] | F1-Score [%] |
|---------------------------------|----------------|------------|--------------|
| U-Net 2D                        | 81             | 89         | 85           |
| U-Net with ResNeXt101 backbone  | 95             | 78         | 86           |
| Attention U-Net                 | 95             | 75         | 84           |


Since the F1-Scores are similar a visual inspection was carried out to find the best-performing model :

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/benchmark.png)

But even clearer is the visual comparison of the running videos:

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/video_results.gif)

## Extracted features

### Time-resolved bubble ratio computation and bubble coverage distribution

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/timeresolved.png)

### Bubble position probability density map

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/heatmaps.png)

### Individual bubble shape analysis

![](https://github.com/andyco98/UTILE-Oxy/blob/main/images/individualcorrect.png)

# *NEW!* UTILE-Oxy Tunnel Vision 3D Spatiotemporal Tracking is Available

With the use of image registration techniques, the extracted segmentation masks are spatiotemporally connected, augmented into a 3D volume and individual bubbles are tracked. 

This opens up new analysis possibilities that weren't available before. Here are some examples:

### Accumulated count of newly emerged bubbles over time

![](https://github.com/alpcanaras/UTILE-Oxy-ALP-FORKED/blob/main/images/New%20Bubbles%20vs.%20Time%201500mV.png)

### Accumulated count of bubble coalescence events

![](https://github.com/alpcanaras/UTILE-Oxy-ALP-FORKED/blob/main/images/Merge%20Events%20vs.%20Time%201500mV.png)

### Denisty map for newly emerging bubble locations

![](https://github.com/alpcanaras/UTILE-Oxy-ALP-FORKED/blob/main/images/Bubble%20Emerging%20Points%20Heatmap%20for%201500mV.png)

### Density map for bubble coalescence locations

![](https://github.com/alpcanaras/UTILE-Oxy-ALP-FORKED/blob/main/images/Bubble%20Merging%20Points%20Density%20Map%20for%201500mV.png)

### Bubble trajectory analysis and mapping

![](https://github.com/alpcanaras/UTILE-Oxy-ALP-FORKED/blob/main/images/trajectory_1500mV_start_end_withavg_and_std_velocity.png)

### Bubble movement metrics extraction

This metrics are calculated and extracted separately for singular (never merging) and merge (that are a results of a coalescence) bubbles and written into a .csv file.

![](https://github.com/alpcanaras/UTILE-Oxy-ALP-FORKED/blob/main/images/Metrics%20CSV.png)

### Bubble coverage lifecycle analysis

This denotes the distribution of coverage duration for related bubbles throughout the video, indicating the time spent in the flow field.

![](https://github.com/alpcanaras/UTILE-Oxy-ALP-FORKED/blob/main/images/lifecycle_histogram_1500mV.png)


## Installation
In order to run the actual version of the code, the following steps need to be done:
- Clone the repository
- Create a new environment using Anaconda using Python 3.9
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
   pip install opencv-python numpy patchify pillow segmentation_models keras tensorflow==2.13.1 matplotlib scikit-learn pandas seaborn tifffile scipy scikit-image pathlib

   ```
### Notes

The datasets used for training and the trained model are available at Zenodo: https://doi.org/10.5281/zenodo.10184579.
