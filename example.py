from skimage import io
import skimage
import numpy as np
import NumpyCNN as npcnn
image = skimage.io.imread(fname = "D://OneDrive//Git_proj//NumpyCNN//WechatIMG1994.jpeg")
image = skimage.color.rgb2gray(image)
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//test.jpeg", arr = image)

l1_filter = np.zeros((2,3,3))

l1_filter[0,:,:] = np.array([[[-1,0,1],
                              [-1,0,1],
                              [-1,0,1]]])
l1_filter[1,:,:] = np.array([[[1,1,1],
                              [0,0,0],
                              [-1,-1,-1]]])

print("开始第一层卷积")
l1_feature_map = npcnn.conv(image, l1_filter)
print(l1_feature_map.shape)
print("开始ReLU")
l1_feature_map_relu = npcnn.relu(l1_feature_map)
print(l1_feature_map_relu.shape)
print("开始池化")
l1_feature_map_relu_pool = npcnn.pooling(l1_feature_map_relu, 2, 2)
print(l1_feature_map_relu_pool.shape)
print("第一层卷积结束")

l2_filter = np.random.rand(3,5,5,l1_feature_map_relu_pool.shape[-1])

print("开始第二层卷积")
l2_feature_map = npcnn.conv(l1_feature_map_relu_pool, l2_filter)
print(l2_feature_map.shape)
print("开始ReLU")
l2_feature_map_relu = npcnn.relu(l2_feature_map)
print(l2_feature_map_relu.shape)
print("开始池化")
l2_feature_map_relu_pool = npcnn.pooling(l2_feature_map_relu, 2, 2)
print(l2_feature_map_relu_pool.shape)
print("第二层卷积结束")

l3_filter = np.random.rand(1,7,7,l2_feature_map_relu_pool.shape[-1])

print("开始第三层卷积")
l3_feature_map = npcnn.conv(l2_feature_map_relu_pool, l3_filter)
print(l3_feature_map.shape)
print("开始ReLU")
l3_feature_map_relu = npcnn.relu(l3_feature_map)
print(l3_feature_map_relu.shape)
print("开始池化")
l3_feature_map_relu_pool = npcnn.pooling(l3_feature_map_relu, 2, 2)
print(l3_feature_map_relu_pool.shape)
print("第三层卷积结束")

skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l1_feature_map.png", arr = l1_feature_map[:,:,0])
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l1_feature_map_relu.png", arr = l1_feature_map_relu[:,:,0])
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l1_feature_map_relu_pool.png", arr = l1_feature_map_relu_pool[:,:,0])

skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l2_feature_map.png", arr = l2_feature_map[:,:,0])
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l2_feature_map_relu.png", arr = l2_feature_map_relu[:,:,0])
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l2_feature_map_relu_pool.png", arr = l2_feature_map_relu_pool[:,:,0])

skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l2_feature_map.png", arr = l2_feature_map[:,:,1])
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l2_feature_map_relu.png", arr = l2_feature_map_relu[:,:,1])
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l2_feature_map_relu_pool.png", arr = l2_feature_map_relu_pool[:,:,1])

skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l2_feature_map.png", arr = l2_feature_map[:,:,2])
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l2_feature_map_relu.png", arr = l2_feature_map_relu[:,:,2])
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l2_feature_map_relu_pool.png", arr = l2_feature_map_relu_pool[:,:,2])

skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l3_feature_map.png", arr = l3_feature_map)
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l3_feature_map_relu.png", arr = l3_feature_map_relu)
skimage.io.imsave(fname = "D://OneDrive//Git_proj//NumpyCNN//l3_feature_map_relu_pool.png", arr = l3_feature_map_relu_pool)