# CS4240 Reproducibility Project: Hierarchical Image Classification using Entailment Cone Embeddings

<p align="center">
 <img src="images/classification_img.PNG" width=300/>
</p>

## Authors

 - Jasper Ketelaar, [j.ketelaar@student.tudelft.nl](j.ketelaar@student.tudelft.nl), 4453050  
 - Aayush Singh, [a.singh-28@student.tudelft.nl](a.singh-28@student.tudelft.nl), 5208122
 
## Introduction

The paper by A. Dhall et al. [1] presents a set of methods for leveraging information about the semantic hierarchy embedded in class labels. They argue that there has been limited work in using unconventional, external guidance other than traditional image-label pairs for training. They model the label-label and label-image interactions using order-preserving embeddings governed by both Euclidean and hyperbolic geometries, prevalent in natural language, and tailor them to hierarchical image classification and representation learning. They empirically validate the models on the hierarchical ETHEC dataset [2]. Here is the image below provided by the authors describing the hierarchy of the dataset:

![ETHEC Sample example](images/ethec_sample.PNG)
_Figure 1: Sample images and their 4-level labels from the ETHEC dataset._

The paper has used different ways to formulate probability distributions to pass hierarchical information like:
- Hierarchy-agnostic baseline classifier (HAB)
- Per-level Classifier
- Marginalization
- Masked Per-level classifier
- Hierarchical Softmax

We aim to reproduce the results shown in the paper by using the existing code (making some changes as required to fit our dataset). First we tried to replicate the results from the ETHEC dataset as used in the paper and then we used a new dataset of wine bottles. Here is the image below describing the hierarchy of our wine dataset:

![Wine Sample example](images/wine_sample.PNG)
_Figure 2: Sample images and their 4-level labels from our wine dataset._

## Datasets

- ETH Entomological Collection (ETHEC) Dataset

The ETHEC dataset used in the paper contains 47,978 images of the “order” Lepidoptera with corresponding labels across 4 different levels. Each image is of size 448x448 pixels. The division of the dataset is like: train(80%), validation(10%) and test(10%) based solely on the images. Here is the image below describing the information mentioned for each image in the json files:

![ETHEC Json example](images/ethec_json.PNG)
_Figure 3: Information for each image in the Json files for the ETHEC dataset, as given by the authors of the paper._

![ETHEC hierarchy example](images/ethec_hierarchy_img.png)
_Figure 4 (paper Figure 1): Hierarchy of labels from the ETHEC dataset across 4 levels: family (blue), sub-family (aqua), genus (brown) and species. For clarity, this visualisation depicts only the first 3 levels. The name of the family is displayed next to its sub-tree. Edges represent direct relations_


[comment]: <> (
--batch_size 32 --experiment_name gcw_exp_3 --experiment_dir gcw --image_dir F:\PycharmProjects\wineset-collector\transforms --n_epochs 4 --model resnet18 --loss last_level --set_mode train --level_weights 1.0 1.0 1.0 1.0 --lr 0.002
--batch_size 32 --experiment_name gcw_exp_2 --experiment_dir gcw --image_dir F:\PycharmProjects\wineset-collector\transforms --n_epochs 4 --model resnet18 --loss hsoftmax --set_mode train --level_weights 1.0 1.0 1.0 1.0 --lr 0.002
)

- Wine Bottle Collection Dataset

The images of the bottles come from a large collection of images that are used to be displayed on the product page
of a webshop. These images are taken from 12 angles to give the user a 360 degree experienece. They are also taken at an 
extremely high resolution after which they are scaled down to different resolutions for zooming in and to make loading times
faster when a high resolution is not required.

We used this dataset for our reproduction because we knew that there were certain levels of hierarchy in the shape, label, the fonts used and more.
To give an example, bottles from the entirety of France usually follow a certain
label structure and even within specific regions there are guidelines that the wineries follow. Moreover, the wineries themselves will often
produce recognizable features within their bottles to create a brand and have people be familiar with it. Most people know what a bottle of
Moët looks like if they are even slightly familiar with wine because their branding has become very popular over the years.

We figured that because these geometries essentially represent a bottle, this could be a very good dataset to work with a geometry based
model in general and upon encountering the paper we actually realized that this could yield fantastic results as it, in essence but obviously more detail,
researched the topic and came up with a multitude of models to be used for data such as ours.

The wine dataset, used to reproduce the results, contains 38893 images with corresponding labels across 4 different levels. 
Each image is kept of the same dimension (448x448 pixels) as the original ETHEC dataset used in the paper. 
The split of the data is also maintained the same: train(80%), validation(10%) and test(10%) based solely on the images. 
Here is the image below describing the information mentioned for each image in the json files:

![Wine Json example](images/wine_json.PNG)
_Figure 5: Information for each image in the Json files for the Wine dataset._

![ETHEC hierarchy example](images/hierarchy_img.png)
_Figure 6: Hierarchy of labels from the Wine dataset across 4 levels: country (blue), region (aqua), winery (brown) and wine. For clarity, this visualisation depicts only the first 3 levels. The name of the country is displayed next to its sub-tree. Edges represent direct relations_

## Methodology

Initially, the existing code was used to replicate results reported in the paper. For executing experiments in different settings certain parameter were required to be modified. We executed experiment in the default parameter setting and some of important parameter worth mentioning are: batch_size = 64, learning rate = 0.00001, optimizer_method=adam, n_epochs=10, weight_strategy = inv, model = resnet50. Apart from these parameters, it had a parameter to choose the loss function (--loss). The options were like:
- 'multi_label' for Hierarchy-agnostic baseline classifier (HAB)
- 'multi_level' for Per-level Classifier
- 'last_level' for Marginalization
- 'masked_loss' for Masked Per-level classifier
- 'softmax' for Hierarchical Softmax

For the wine dataset, some changes were required to the existing code, json files had to be generated for train and test purposes, and image transformations were required to create some noise in the image and to make the dataset comparable in size to the ETHEC dataset used in the paper. Firstly, from the entire wine dataset only those data points were collected were there were images available and the parents in the hierarchy had some minimum number of children. Also, six different angles of wine bottle images were selected. After downloading the images as per the decided criteria, these images were transformed. Originally images were of different size, so these images were scaled up or down to 448x448 pixels, as per requirement. Then, the images were introduced to some gaussian noise and the changes can be seen in the images below:

<p align="center">
 <img src="images/wine_no_noise.PNG" width=300>
 <img src="images/wine_noise.PNG" width=300>
</p>

_Figure 7: Representation of Wine image after transformation._

## Results

The results presented below are our effort to reproduce the paper. The original results from are also mentioned as a reference, along with the results obtained from our experiments on the ETHEC data set and the Wine dataset. The scores shown in the table are micro-averaged F1 scores which is calculated as F1 = (2*P*R)/(P + R) (with P being the precision and R being recall). A micro-averaged score for a metric is calculated by accumulating contributions (to the performance metric) across all labels and these accumulated contributions are used to calculate the micro score.

| | m-F1 | L1 | L2 | L3 | L4 |   
| --- | --- | --- | --- | --- | --- |
| ETHEC dataset | --- | --- | --- | --- | --- |
| **HS (Paper)** | 0.9180 | 0.9879 | 0.9731 | 0.9253 | 0.7855 |   
| **HS (Our Result)** | 0.6427 | 0.7316 | 0.7230 | 0.6927 | 0.5892 |
| **MC (Paper)** | 0.9223 | 0.9887 | 0.9758 | 0.9273 | 0.7972 |
| **MC (Our Result)** | 0.6061 | 0.6807 | 0.6749 | 0.6136 | 0.4801 |
| Wine dataset | --- | --- | --- | --- | --- |
| **HS** |  |  |  |  |  |
| **MC** |  |  |  |  |  |

_Table 1: Results obtained from experiments._

Looking at the results for the ETHEC dataset, we concluded that the setting in which we ran the experiments were not appropriate. We believe the low scores in comparison to those in paper is mainly due limited number of epochs used in our experiments. Due to high execution time and memory consumption, we had to limit the number of epochs to 10.

TODO: Mention something about our dataset results.

## Conclusion



Apart from the reproduced results we would like to conclude with a remark about the overall reproducibility of the paper. We realized that the available code repository didn't have a proper readme.md file nor there were proper comments in the code. This made it difficult to understand and manipulate the code. We also contacted the author of the paper regarding some issue while executing the order-embedding code, but did not get much information as the author was not very relevant with the code at the time. There were also some issues related to execution time and memory consumption. This limited our experiments to less number of trials and made it difficult to validate our results.

## References
- \[1\]: Ankit Dhall, Anastasia Makarova, Octavian Ganea, Dario Pavllo, Michael Greeff, & Andreas Krause (2020). Hierarchical Image Classification using Entailment Cone Embedding. 

- \[2\]: A. Dhall (2019), Eth entomological collection (ethec) dataset  https://www.researchcollection.ethz.ch/handle/20.500.11850/365379 .

- \[3\]: A. Dhall (2019). Learning Representations For Images With Hierarchical Labels.

## Work Division

During the entire course of this project, we worked in a collaborative manner. Initially, we had discussions regarding which paper to reproduce and what dataset should we utilize. Then we focused on understanding the original paper and had discussions among ourselves and with the teaching assistant. Below the task division between the group members is shown:

### Jasper Ketelaar
- First Point

### Aayush Singh
- First Point


