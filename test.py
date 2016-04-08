import matplotlib.pyplot as plt

from skimage import data, io
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity


ihc_rgb = data.immunohistochemistry()
ihc_hed = rgb2hed(ihc_rgb)

h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1))

# io.imshow(ihc_hed[:, :, 0], cmap=plt.cm.gray)
io.imshow(h)

io.show()
