import imageio
import numpy as np
import caffe
import os
import skimage.color as color
import scipy.ndimage.interpolation as sni
import commands
import os
import subprocess
import sys

if len(sys.argv) < 5:
    print ("Usage %s inputimage outputimage proto model" % sys.argv[0])
    sys.exit(1)

inputimage = sys.argv[1]
outputimage = sys.argv[2]
proto = sys.argv[3]
model = sys.argv[4]

net = caffe.Net(proto,model, caffe.TEST)
#net = caffe.Net('/home/luluf/colorization/models/colorization_deploy_v2.prototxt', 1, weights='/home/luluf/colorization/models/colorization_release_v2.caffemodel')

(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape
net.blobs['Trecip'].data[...] = 6/np.log(10) # 1/T, set annealing temperature
    # (We found that we had introduced a factor of log(10). We will update the arXiv shortly.)
    
# load the original image
#img_rgb = caffe.io.load_image('/mnt/Nanterre/Lulu/Projets/Colorisation/raycharles/original/original.4661.png')
img_rgb = caffe.io.load_image(inputimage)
img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
img_l = img_lab[:,:,0] # pull out L channel
(H_orig,W_orig) = img_rgb.shape[:2] # original image size

# resize image to network input size
img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
img_lab_rs = color.rgb2lab(img_rs)
img_l_rs = img_lab_rs[:,:,0]

net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
caffe.set_mode_cpu()
net.forward() # run network

ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
img_rgb_out = np.clip(color.lab2rgb(img_lab_out),0,1) # convert back to rgb

#print ("network size : %d %d",H_in,W_in)
imageio.imsave(outputimage, img_rgb_out)
#imageio.imsave("tmp.png", img_rgb_out)
