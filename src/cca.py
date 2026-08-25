from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localisation

# label the connected regions of the binary image
label_image = measure.label(localisation.binary_car_image)
fig, (ax1) = plt.subplots(1)
ax1.imshow(localisation.gray_car_image, cmap="gray")

# regionprops exposes the geometry of each labelled region
for region in regionprops(label_image):
    if region.area < 50:
        # too small to be a plate
        continue

    # outline the region on the source image
    minRow, minCol, maxRow, maxCol = region.bbox
    rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
    ax1.add_patch(rectBorder)

plt.show()