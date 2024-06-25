ground_points = las.points[las.number_of_returns == las.return_number]

print("%i points out of %i were ground points." % (len(ground_points),
        len(las.points)))

#why does this create a different value for ground points in lidat.ipynb? 