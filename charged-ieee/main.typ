#import "@preview/charged-ieee:0.1.0": ieee 
#import "@preview/codly:0.2.1": *

#show: ieee.with(
  title: [A introduction to Data Mining and its applications with Point Clouds],
  abstract: [
    In this paper, we focus on the necessary steps in Data Mining and how this process is applied with Point Clouds in Geospatial Data.
    We will shine light on several steps of a Data Mining Pipeline, including Data Cleaning, Feature Enginnering followed by model selection and 
    important aspects that have to be considered when training the selected model. 
    
  ],
  authors: (
    (
      name: "Matthias Weil",
      department: [School of Computation, Information and Technology, Informatics],
      organization: [Techincal University Munich],
      location: [Munich, Germany],
      email: "matthias.weil@tum.de"
    ),
  ),
  index-terms: ("Data Mining", "Geospatial Data", "Point-Clouds", "3D","Feature Extraction","Classification","Co-Registration"),
  bibliography: bibliography("refs.bib"),
)

= Introduction
Data Mining and extracting data from Point Clouds are necessary skills when working with Geospatial Data sets. Many methods and how they can be practically applied are explained.
= Data Mining 
== Data Cleaning
It is not uncommon for the data obtained to be of poor quality and difficult to work with. This may include the presence of missing values, outliers, or spelling mistakes, which are particularly prevalent in user-generated data.

== Feature Extraction
Machine learning algorithms require the given data to be in certain numerical formats. In this step, it is necessary to transform the data into the needed format to allow further processing. Additionally, this allows the machine learning algorithm to receive a more effective set of inputs, which increases accuracy and lowers the computational need for machine learning algorithms.

=== Over- and Underfitting
#figure(
  image("images/overUnderfitting.png",width:70%),
  caption: [A common problem when working with a machine learning algorithm is to create a model that discovers patterns in unknown data. On the left is Underfitting, when the model is too simple to capture the complexity of the data. In the middle is the desired outcome of a data model. On the right is the example of Overfitting which occurs when a machine learning algorithm detects noise as data. Figure from @overfitting.]
)
== Feature Selection <feature>
Feature selection is a critical part of data mining due to numerous reasons. As only relevant features are being considered, the dimensionality of the data set is reduced and is more efficient, accurate, and less prone to overfitting. Furthermore less features lead to a better understanding of the learning result. The following figure explains the Hughes phenomenon, where a higher amount of features deteriorates the model's performance. Additional features may introduce noise, which leads to a decrease in classification accuracy @hughes. 
#figure(
  image("images/hughesScreen.png",width:70%),
  caption:[Hughes phenomenon, a increase of number of features decreases the accuracy of the classification model. Figure from @WEINMANN2015286.]
)

There are three notable categories of feature selection. Filter-based methods include statistical tests to assess the correlation between various features. These filters for example remove features with low variance, since they are deemed to have little information. Because they are classifier-independent, simple, and efficient they are often used. Alternatively, Wrapper-based methods or Embedded methods can be used @Weinmann2016.
== Model Selection 
Two overarching themes emerge when selecting a model: supervised and unsupervised learning models. Supervised machine learning includes classification which is heavily used in image recognition. Examples of classification models that categorize data into labels, are Decision Trees, Support Vector Machines, and k-nearest Neighbors. Supervised learning also includes regression models which describe functions that model the correlation of an independent variable and a target variable. The correlation between systolic blood pressure and the amount of coffee a person drinks is an example of a linear regression.

In unsupervised classification, the algorithm tries to assign each data point to a cluster. The number of clusters to be extracted depends on the algorithm. In the case of kMeans, the user must input the desired number of clusters. Alternatively, in the case of DBSCAN, the user must define the distance between two points to form a new cluster. These algorithms work without training labels, so the user has to understand the output clusters.

Deep learning as the machine learning algorithm called representation learning harvests the benefit, of the algorithm which learns the features from the data itself. The algorithm is then capable of representing the data in a way that eases classification or regression @ETRAINEE.
== Model Training
Models are prone to be overfitted when too many features are selected, sufficient data not being available to train the model, or if the data is noisy. This noise may easily be detected as a pattern by the model. To detect overfitting the data set is split into a training and a validation set. After training, the model is presented with the validation set, and the performance is tracked. If they diverge, then it can be concluded that the model is overfitting. To tackle underfitting additional features can be selected, and the complexity of the model can be raised or an increase of the duration of training the model @SCIKITLEARN.
== Model Evaluation 
To evaluate the model various methodologies may be applied. For regression, Mean Absolute Error or Mean Squared Error is often applied. To evaluate a classification model one can calculate the Accuracy which is the proportion of correctly classified instances. Furthermore, the Precision, which measures the quality of positive predictions, Recall, the ability of the classifier to find all the positive samples, and the F1-score, which is the Harmonic mean of precision and recall, illustrate various metrics to evaluate a classifier @SCIKITLEARN @MANNING2009.


= Geospatial Data Mining
In the following, we showcase a typical workflow for classification that involves first having a point cloud that can be expanded by co-registration. Then neighborhood selection, followed by feature extraction and feature selection. Lastly, we choose our classification model. 
== Co-Registration
When working with point cloud data there may be two acquisitions from the same structure, therefore it is beneficial to apply Co-registration to combine the multiple point clouds of interest. Typical algorithms include iterative closest point algorithm or feature based matching algorithms. 
#figure(
  image("images/coRegistration.png",width: 70%),
  caption:[Co-registration of two point clouds, showing the Arc de Triomphe from different angles. Here a feature based matching algorithm, along the corners of the building, might be more computationally efficient. Figure from @registration.]
)<co>

As can be seen in @co, with two point clouds of the Arc de Triomphe in Paris, the co-registered point cloud has more information about the subject of interest. Often co-registration is conducted to detect changes in the area, for example after natural disasters or after longer periods to show change in the scanned area.

#figure(
  image("images/screenshot_000001.png",width: 70%),
  caption: [Two point cloud scans of the hellstugubrean glacier visualized using python.The point cloud from 2009 showing the intensity is colored in blue and the same glacier from 2017 colored in yellow. Figure by author. ]
)<glacier>

@glacier illustrates how Co-registration is a vital procedure when working with point-cloud data. @ETRAINEE notes that the difference in height between the two point clouds, which can be seen most clearly in the center of the image, is due to a "significan[t] loss of mass from the glacier over these 8 years". 

== Neighborhood Selection
Calculations based on neighborhoods are among the most important ones, to gain additional information. They are necessary for any filtering, smoothing, or interpolation step and information extraction.

Spatial neighborhoods are defined either by a certain distance to other points or to a certain amount of fixed neighbors @WEINMANN2015286.
If a fixed distance is given, and the height can be neglected, the neighborhood is commonly referred to as a cylindrical neighborhood. Otherwise, if it‘s 3D then they are called spherical neighborhoods because their height is also relevant. A point cloud neighborhood can also be defined by a fixed amount of neighbors. Fixed neighbors are also commonly called k-nearest neighbors and the size can be highly variable depending on the point density at that area @ETRAINEE.


#figure(
    grid(
        columns: (auto, auto),
        rows:    (auto, auto),
        gutter: 1em,
        [ #image("images/cylindrical.png",   width: 100%) ],
        [ #image("images/kNN.png", width: 100%) ],
    ),
    caption: [Spatial and k-nearest neighbor neighborhoods shown side by side. Figures by @ETRAINEE.]
) <neighborhood>

== Feature Extraction
After neighborhoods are defined, various features can be extracted. 3D-features can be calculated by using eigenvalues or looking at the geometric properties. Spatial neighborhoods allow the derivation of local surface roughness.
#figure(
  image("images/KitFeatures.png"),
  caption:[Features that can be easily derived from eigenwerte. Figure and more details in @WEINMANN2019.]
)

The following code provided by @ETRAINEE demonstrates how a feature vector can be created, focusing on extracting features that can be calculated by using eigenwerte. They include planarity, linearity, omnivariance, roughness, and slope. These features are calculated using eigenvalues derived from the covariance matrix of neighboring points. There are a multitude of other features that can be extracted @Weinmann2016.

#show: codly-init.with()
#codly(
  
)
```python
def create_feature_vector(neighbour_points):
    # structure tensor
    struct_tensor = np.cov(neighbour_points.T)
    # eigenvalue decomposition
    eigvals, eigvec = np.linalg.eigh(struct_tensor)
    l3, l2, l1 = eigvals

    # find eigenvector to smallest eigenvalue = normal vector to best fitting plane
    normalvector = eigvec[:, 0]
    # flip so that it always points "upwards"
    if normalvector[2] < 0:
        normalvector *= -1

    # feature calculation
    planarity = (l2-l3)/l1
    linearity = (l1-l2)/l1
    omnivariance = (l1*l2*l3)**(1./3)
    roughness = l3
    slope = np.arctan2(np.linalg.norm(normalvector[:2]), normalvector[2])

    return np.array([planarity,
                     linearity,
                     omnivariance,
                     roughness,
                     slope])
```
=== Ground Removal
Ground removal is an important procedure when working with point cloud data. Depending on the specific situation, it might be beneficial or even necessary to remove ground points. Sometimes it is cost-effective to remove them to reduce computing power and time needed to process the data.
Notable algorithms for this purpose are RANSAC (Random Sample Consensus), Ground Plane Fitting, and Patchwork++.

#figure(
  image("images/groundRemoval.png",width: 70%),
  caption: [
    Showcasing different ground removal algorithms and Patchwork++ being shown as least prone to under-segmentation (blue area). _Green_: True Positive _Blue_: False Negatives _Red_: False Positives. Figure from @lee2022patchwork.
  ]
)
/*
The next section demonstrates how the number of ground points can be extracted from the given dataset. For the glacier data, ground removal is underwhelming because a glacier inherently only contains ground points. So for this example, we use a dataset we extracted from @NASAGLIHT.

If you remember the las files, their specification allows points to be classified to certain values. For example: 
1. Unassigned
2. Ground 
3. Low vegetation 

For more details please look here @OGCLAS and for visual aid @ARCGISLIDAR. 

The las files have already run through a classification pipeline, easing further work on them. This allows us to filter points just by specifying the classification value which indicates the type of surface the LiDAR pulse has hit. But not all point clouds have all categories classified. The data from G-LiHT is only split into unassigned and ground points for example. Otherwise, it is possible to get a broad estimate by using the amount of returns, and the order of reflections a LiDAR pulse encounters as it travels through the environment.

#show: codly-init.with()
#codly(
  
)
```python
pub fn main() {
    println!("Hello, world!");
}
```
As we can see there is a slight discrepancy between the amount of ground points, that may be due to the return number being a simpler approach. Using the classification that has already gone through a whole pipeline, improves accuracy and leads to a more exact value.
*/

== Feature Selection
For a detailed explanation of feature selection, please refer back to @feature. Next, we show an example of a filter based method to select valuable features. The following code snippet, taken from @SCIKITLEARN is designed to remove features "that are either one or zero in more than 80% of the samples.", according to scikit-learn documentation. As they are boolean features the threshold is given by
$ "Var"[X] = p(1 - p) $
so the correct threshold for the function is
 $.8 * (1 - .8) $

 #show: codly-init.with()
#codly(languages:(
  python: (name: "Python", color: rgb("#3572A5")),
),
  stroke-width:0.5pt,
  stroke-color: red,
  display-icon: false,
)
```python
from sklearn.feature_selection import VarianceThreshold

X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

sel.fit_transform(X)
```
== Classification 
The extracted features can then be used as input for classifiers that have been trained with representable data. Noteworthy classifiers include Random Forest, Support Vector Machines, Nearest Neighbor classifiers, Decision Trees, Naïve bayesian classifiers and Linear Discriminant Analysis.
