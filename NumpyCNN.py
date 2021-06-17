import numpy as np
import sys

from numpy.core.records import array

def augmentMatrix(A, b):    #矩阵增广
    return [AA + bb for AA, bb in zip(A,b)]

def conv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))

    for r in np.uint16(np.arange(filter_size/2.0,
                                img.shape[0]-filter_size/2.0+1)): #宽640
        for c in np.uint16(np.arange(filter_size/2.0,
                                    img.shape[1]-filter_size/2.0+1)): #长1024
            curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)),
                              c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))] #以某一点为中心选出一个框
            curr_result = curr_region * conv_filter
            #im2col = curr_region[i]
            conv_sum = np.sum(curr_result)
            result[r,c] = conv_sum
    final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0),
                          np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
    return final_result

def optConv_(img, conv_filter):
    filter_size = conv_filter.shape[1]
    acculateImg = np.zeros(((img.shape[0]-filter_size+1)*(img.shape[1]-filter_size+1),conv_filter.shape[1]**2))
    count = 0
    for r in np.uint16(np.arange(filter_size/2.0,
                            img.shape[0]-filter_size/2.0+1)): #宽640
        for c in np.uint16(np.arange(filter_size/2.0,
                                    img.shape[1]-filter_size/2.0+1)): #长1024
            curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)),
                              c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
                               #以某一点为中心选出一个框
            #print(curr_region)
            temp = np.zeros(len(curr_region)*len(curr_region[0]))
            tempCnt = 0
            for r in range(len(curr_region)):
                for c in range(len(curr_region[r])):
                    temp[tempCnt] = curr_region[r,c]
                    
            acculateImg[count] = temp
            count += 1
    return acculateImg

def optConv(img, conv_filter):

    if len(img.shape) != len(conv_filter.shape) - 1:
        print("错误：图片或者卷积过滤器不正确。")
        exit()
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print("错误：图片与卷积过滤器通道数必须匹配。")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print("错误：卷积过滤器必须为正方形矩阵。")
        sys.exit()
    if conv_filter.shape[1]%2 == 0:
        print("错误：卷积过滤器必须为奇数。")
        sys.exit()

    filter_size = conv_filter.shape[1]
    result = np.zeros(((img.shape[0]-filter_size+1)*(img.shape[1]-filter_size+1),conv_filter.shape[0]))
    final_result = np.zeros((img.shape[0]-filter_size+1,img.shape[1]-filter_size+1,conv_filter.shape[0]))

    if len(conv_filter.shape) > 3:
        acculateFilter = np.zeros((conv_filter.shape[1]**2 * conv_filter.shape[-1],conv_filter.shape[0]))
        acculateImg = np.zeros(((img.shape[0]-filter_size+1)*(img.shape[1]-filter_size+1),conv_filter.shape[1]**2 * conv_filter.shape[-1]))
        for map_num in np.arange(0,conv_filter.shape[0]):
            tempi = 0
            for z in np.arange(0,conv_filter.shape[3]):
                for r in np.arange(0,conv_filter.shape[1]):
                    for c in np.arange(0,conv_filter.shape[2]):
                        acculateFilter[tempi,map_num] = conv_filter[map_num, r, c, z]
                        tempi += 1
        sumAcculateImg = optConv_(img[:,:,0],conv_filter)
        for z in np.arange(1,img.shape[2]):
            acculateImg = optConv_(img[:,:,z],conv_filter)
            sumAcculateImg = augmentMatrix(sumAcculateImg,acculateImg)
        result = acculateImg.dot(acculateFilter)
        
    else:


        acculateFilter = np.zeros((conv_filter.shape[1]**2,conv_filter.shape[0]))
        acculateImg = np.zeros(((img.shape[0]-filter_size+1)*(img.shape[1]-filter_size+1),conv_filter.shape[1]**2))
        
        for map_num in np.arange(0,conv_filter.shape[0]):
            tempi = 0
            for r in np.arange(0,conv_filter.shape[1]):
                for c in np.arange(0,conv_filter.shape[2]):
                    acculateFilter[tempi,map_num] = conv_filter[map_num, r, c]
                    tempi += 1

        acculateImg = optConv_(img,conv_filter)
        
        result = acculateImg.dot(acculateFilter)

    tempII = 0
    for num_map in np.arange(0,conv_filter.shape[0]):
        for r in np.arange(0,final_result.shape[0]):
            for c in np.arange(0,final_result.shape[1]):
                final_result[r,c,num_map] = result[tempII,num_map]
                tempII += 1


    return final_result



def conv(img, conv_filter):

    if len(img.shape) != len(conv_filter.shape) - 1:
        print("错误：图片或者卷积过滤器不正确。")
        exit()
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print("错误：图片与卷积过滤器通道数必须匹配。")
            sys.exit()
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print("错误：卷积过滤器必须为正方形矩阵。")
        sys.exit()
    if conv_filter.shape[1]%2 == 0:
        print("错误：卷积过滤器必须为奇数。")
        sys.exit()
    
    feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1]+1,
                            img.shape[1]-conv_filter.shape[1]+1,
                            conv_filter.shape[0]))

    for filter_num in range(conv_filter.shape[0]):
        print("Filter ", filter_num + 1)
        curr_filter = conv_filter[filter_num, :]


        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0])
            for ch_num in range(1, curr_filter.shape[-1]):
                conv_map = conv_map + conv_(img[:,:,ch_num], curr_filter[:,:,ch_num])
        else:
            conv_map = conv_(img, curr_filter)
        feature_maps[:,:,filter_num] = conv_map
    return feature_maps

def pooling(feature_map, size = 2, stride = 2):
    pool_out = np.zeros((np.uint16((feature_map.shape[0]-size+1)/stride+1),
                        np.uint16((feature_map.shape[1]-size+1)/stride+1),
                        feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0,feature_map.shape[0]-size+1, stride):
            c2 = 0
            for c in np.arange(0,feature_map.shape[1]-size+1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r:r+size, c:c+size, map_num]])
                c2 = c2 + 1
            r2 = r2 + 1
    return pool_out

def relu(feature_map):
    relu_out = np.zeros(feature_map.shape)
    for r in np.arange(0,feature_map.shape[0]):
        for c in np.arange(0,feature_map.shape[1]):
            for map_num in range(feature_map.shape[-1]):
                relu_out[r,c,map_num] = np.max([feature_map[r,c,map_num], 0])
    return relu_out