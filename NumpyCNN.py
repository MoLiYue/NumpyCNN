from skimage import io
import skimage

image = skimage.io.imread(fname = "D:\OneDrive\Git_proj\NumpyCNN\WechatIMG1994.jpeg", as_grey = True)
skimage.io.imsave(fname = "D:\OneDrive\Git_proj\NumpyCNN\test.jpeg", arr = image)

