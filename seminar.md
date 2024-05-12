# point-based approach
Process point clouds to produce a sparse representation 
Extraction of a feature vector by aggregating their neighbouring features 
feature examples: Local, Global or Contextual features 
local = rich in detailed information about localisation of a point 
global = encode geometric structure for a point and its neighbours 
contextual = latest stage of pipeline, rich in localisation and semantic information 
# data science problems
## ground removal
## plane fitting
pcfitplane 
fitted plane is pronte to be influenced by noises. 
## (co-) registration 
## segmentation 
classyifing into multiple homogenours regions, similar properties. 
challenges: high redundancy, uneven smapling density, no explicit structure of poinnt cloud data 
https://paperswithcode.com/task/point-cloud-segmentation
These are the major approaches for point cloud semantic segmentation.

Classify each point or a point cluster based on individual features by using feature extraction and neighborhood selection.

Extract point statistics and spatial information from the point cloud to classify the points using statistical and contextual modeling.
### Deep Learning approach 
https://de.mathworks.com/help/lidar/ug/sematic-segmentation-with-point-clouds.html
cant apply usual CNN because of unordered, sparse, and unstructured nature of point cloud data -> transformation of raw point clouds necessary. 
3 different approaches: 
multiview-based 
Voxel-based 
Point-based (Point-Net)
## classification 
## object detection 
combination of LIDAR and images for accuracy of small objects.

Das pointnet thema wird in einem anderen talk gemacht 

# PointNET 
## Weaknesses of pointnet 
 Although this network can be applied for extraction of geometric features and object classification purposes [37], it has serious limitations in capturing local structure information between neighbouring points, since features are learned individually for each point and the relation between points is ignored 
## PointNet++

ppt-> 15 min data mining erklärung, schließlich besonderheiten für geospatial data -> jeweils code/ bild examples (layout mit 2 componenten)


ppt: masteransicht, layout, rahmen kllicken für auto layout, 

ibm geospatial data was used to predict crop and soy yield. 
-> make big monies if you use in june when US agriculture says yield -> price increase, decrease


# data Mining

## KDD Process 
Selection, Preprocessing, Transformation, Data Mining, Interpretation 

## classification model 
binary or nominal answers -> the former is easier to achieve 
## six steps to build a classification model 


## data Collection 1.
relevant data should be collected, from database or similar 
## preprocessing 2.
handling missing values, -> replacing with mean, median 
 dealing with outliers, -> removal or replacing as above. can be found with statistical methods 
  suitable format, -> normalization 
   numerical format 
## feature selection 3.
identify most relevant attributes 
highly correlated features can be removed because they dont provide additional information 
high information gain features are selected 
### PCA 
## model selection 4. 
### decision trees 
### SVM 
### NN 
## model training 5. 
data divided in training and validations set 
## model evaluation 6. 
evaluation of the model 