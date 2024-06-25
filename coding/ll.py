from pygroundsegmentation import GroundPlaneFitting
import numpy as np

ground_estimator = GroundPlaneFitting() #Instantiate one of the Estimators

xyz_pointcloud = np.random.rand(1000,3) #Example Pointcloud
ground_idxs = ground_estimator.estimate_ground(xyz_pointcloud)
ground_pcl = xyz_pointcloud[ground_idxs]

