import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cca2


# invert so the characters become the white foreground that measure.label expects
license_plate = np.invert(cca2.plate_like_objects[0])

labelled_plate = measure.label(license_plate)

fig, ax1 = plt.subplots(1)
ax1.imshow(license_plate, cmap="gray")
# character size bounds relative to the plate: 60-80% of its height, 5-15% of its width
character_dimensions = (0.6*license_plate.shape[0], 0.8*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

characters = []
column_list = []
for regions in regionprops(labelled_plate):
    y0, x0, y1, x1 = regions.bbox
    region_height = y1 - y0
    region_width = x1 - x0

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1]

        # outline the character on the plate
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                       linewidth=2, fill=False)
        ax1.add_patch(rect_border)

        # normalise to 20x20, the input size the classifier was trained on
        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        # record the x-position so the characters can be reordered later
        column_list.append(x0)

plt.show()