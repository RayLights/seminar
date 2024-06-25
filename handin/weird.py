import laspy
import numpy as np

# Load the LAS file
las = laspy.read('Wertheim_Jun2016_c1r0.las')

# Define constants
GROUND_RETURN_NUMBER = 2
GROUND_CLASSIFICATION_VALUE = 2

# Extract coordinates
coords = np.vstack((las.x, las.y, las.z)).transpose()

# Extract attributes
return_numbers = las.return_number
classification_values = las.classification

# Filter ground points using the return number
ground_mask_return = return_numbers == GROUND_RETURN_NUMBER
coords_ground_return = coords[ground_mask_return]

# Filter ground points using the classification value
ground_mask_classification = classification_values == GROUND_CLASSIFICATION_VALUE
coords_ground_classification = coords[ground_mask_classification]

# Number of ground points
num_ground_points_return = len(coords_ground_return)
num_ground_points_classification = len(coords_ground_classification)

# Total number of points
total_points = len(coords)

# Calculate percentages
percent_ground_points_return = (num_ground_points_return / total_points) * 100
percent_ground_points_classification = (num_ground_points_classification / total_points) * 100

# Print the number of ground points and percentages
print(f"We kept {num_ground_points_return} ground points using return number out of {total_points} total ({percent_ground_points_return:.2f}%)")
print(f"We kept {num_ground_points_classification} ground points using classification value out of {total_points} total ({percent_ground_points_classification:.2f}%)")

# Grab the return_num and num_returns dimensions
ground_points = las.points[las.number_of_returns == las.return_number]

print("%i points out of %i were ground points." % (len(ground_points),
        len(las.points)))

print("why do all of these return different values? how to do it better?")