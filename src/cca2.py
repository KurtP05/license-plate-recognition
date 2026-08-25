from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localisation

# label the connected regions of the binary image
label_image = measure.label(localisation.binary_car_image)

# plate size bounds, expressed as fractions of the image dimensions
plate_dimensions = (0.06*label_image.shape[0], 0.10*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []
fig, (ax1) = plt.subplots(1)
ax1.imshow(localisation.gray_car_image, cmap="gray");

# regionprops exposes the geometry of each labelled region
for region in regionprops(label_image):
    if region.area < 50:
        # too small to be a plate
        continue

    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
    # keep regions sized like a plate and wider than they are tall
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        plate_like_objects.append(localisation.binary_car_image[min_row:max_row,
                                  min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                              max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
        ax1.add_patch(rectBorder)

plt.show()