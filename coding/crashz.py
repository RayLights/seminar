# %% [markdown]
# # Point-Cloud-Data Workflow

# %%
import laspy
import numpy as np
import open3d as o3

# %% [markdown]
# #### Importing our Point-Cloud-Data
# This data is imported from [G-LiHT](https://glihtdata.gsfc.nasa.gov/)\
# And was then converted to .las format.\
# .laz files can be used if laztools or lazrs are installed as backend. 

# %%
las = laspy.read('/Users/Elaji/seminar/las_data/Wertheim_Jun2016_c1r0.las')
list(las.point_format.dimension_names)

# %%
las.X
las.intensity
las.gps_time

# %% [markdown]
# ### PCD Classifications
# With LAS 1.1-1.4 Specification there are defined categorys. 
# 1. Unassigned
# 2. Ground
# 3. low Vegetation
# 6. Building
# 9. Water\
# [More Details](https://desktop.arcgis.com/en/arcmap/latest/manage-data/las-dataset/lidar-point-classification.html)

# %%
#set(list(las.classification))

# %%
non_ground = las.points[las.classification==1]
point_data = np.stack([non_ground.X,non_ground.Y,non_ground.Z],axis=0).transpose((1,0))
geom = o3.geometry.PointCloud()

# %%
geom.points = o3.utility.Vector3dVector(point_data)


# %%
o3.visualization.draw_geometries([geom])


