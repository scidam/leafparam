

from skimage.io import imread
from skimage import feature
from skimage.color import rgb2gray
from skimage import measure

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn import cross_validation

from skimage.morphology import  white_tophat
from skimage.morphology import  disk
from skimage.transform import resize

from skimage.segmentation import slic


from sklearn.cluster import KMeans
import pdb
# 1. image filtering... 
# 2. 


RESOLUTION = float(3000)

def is_closed(contour):
  return (contour[0]==contour[-1]).all()


def length(contour):
  return np.sum(np.linalg.norm(np.array(contour[1:])-np.array(contour[:-1]),axis=1))
  

def get_largest(contours):
    _l = [length(x) for x in contours]
    return contours[np.argmax(_l)]


def parametrize(contour):
    _cont = np.array(contour)
    tck,u=interpolate.splprep([_cont[:,0],_cont[:,1]],s=0)
    unew = np.arange(0,1.0+1.0/RESOLUTION,1.0/RESOLUTION)
    r = np.array(interpolate.splev(unew,tck)).transpose()
    dr = np.array(interpolate.splev(unew,tck,der=1)).transpose()
    _t = np.arctan(dr[:,1]/dr[:,0])
    _t[dr[:,0]<0.0] = np.pi+np.arctan(dr[dr[:,0]<0.0,1]/dr[dr[:,0]<0.0,0])
    _l = [length(r[:j]) for j in range(1,len(r)+1)]
    return (_t, _l, r)


def getpoints(t,l):
  points=[[0,0]]
  for ind,k in enumerate(t[1:-1]):
    points.append(list(np.array(points[-1])+np.array([np.cos(k), np.sin(k)])*(l[ind+1]-l[ind])))
  return np.array(points)


def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    ax1.imshow(original)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered,cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')


def leaf_image_preprocess(image, maxdimension=700):
    '''
    Resize image to max dimension 700.
    Image assumed to be an RGB-image with dimension: mxnx3.
    
    '''
    a, b, c = np.shape(image)
    maxdim = max(a,b)
    ratio = float(a)/float(b)
    if a == maxdim:
        _a = float(maxdimension)
        _b = _a/ratio
    else:
        _b = float(maxdimension)
        _a = ratio*_b

    return resize(image, (int(_a),int(_b)))
       
       
  

def extract_leaf_stem(image, maxdisksize=30, minstempixels=100):
    width,height,depth = np.shape(image)
    X = image.reshape(width*height,depth)
    clast = KMeans(n_clusters=2, n_jobs=-2)
    clast.fit(X)
    lbls = np.array(clast.labels_).reshape(width,height)
    if np.mean(lbls[width/2-10:width/2+10,height/2-10:height/2+10])>0.5:
        rightlbl=1
    else:
        rightlbl=0
    lbls = (lbls==rightlbl).astype(np.int16)
    im_flt=white_tophat(lbls,disk(maxdisksize))
#     cnts=measure.find_contours(im_flt, 0.8)
    flb=measure.label(im_flt,connectivity=2) 
    props=measure.regionprops(flb)
    cc = [item['inertia_tensor_eigvals'][0]/item['inertia_tensor_eigvals'][1] if item['inertia_tensor_eigvals'][1]>0.0 and item['area']>=minstempixels else 0.0 for item in props]
    res =  lbls.copy()
    if any(cc):
        indmax = np.nanargmax(cc)
        stem  = (flb==indmax+1)
        res[stem]=0
    return res





#----------------------------------------------------------- 





  
from files import dataset
labels = np.array(dataset.keys(), dtype=np.int)



allpars = []
alllen = []
alllab = []
graph=[]


# lbls = slic(im, n_segments=2,compactness=1)
 
# im = imread(dataset[labels[0]][4])
# lbls = extract_leaf_stem(im)
# plot_comparison(im, lbls, 'sdf')
# plt.show()
# sdf




# img_gray = rgb2gray(im)
# img_gray = resize(img_gray,(500,500))
#  
#  
# lbls = slic(im, n_segments=2,compactness=1)
#  
# im_flt=white_tophat(np.invert(lbls),disk(30))
#  
# cnts=measure.find_contours(im_flt, 0.8)
#  
# flb=measure.label(im_flt,connectivity=2)
#  
# props=measure.regionprops(flb)
#  
# cc = []
# for ind,item in enumerate(props):
#   print ind, item['inertia_tensor_eigvals']
#   if item['inertia_tensor_eigvals'][1]>0:
#     cc.append(item['inertia_tensor_eigvals'][0]/item['inertia_tensor_eigvals'][1])
#   else:
#     cc.append(0.0)
#  
# ind = np.nanargmax(cc)
# print ind, cc[ind]
#  
# stem  = (flb==ind+1)


for k in labels:
  for j in dataset[k]:
    im = imread(j)
    im = leaf_image_preprocess(im)
    img_filt = extract_leaf_stem(im)
    contours = measure.find_contours(img_filt, 0.8)
    a,b,c=parametrize(get_largest(contours))
    allpars.append(np.abs(np.fft.fft(a))/float(len(a)))
    alllen.append(b)
    alllab.append(k)
    graph.append(a)
    print j
    
   
#print np.shape(alllab), np.shape(allpars)

#lda = LDA()

#print cross_validation.cross_val_score(lda, allpars, alllab, cv=4),len(graph)

fig = plt.figure()
ax = fig.add_subplot(111)





for k in [8, 1]:
    #ffts = np.fft.fft(graph[k])
    #freq = np.fft.fftfreq(np.shape(graph[k])[-1])
    #print np.max(freq), np.min(freq)
    #ffts[freq>0.2]=0.0
    pts = getpoints(graph[k], np.array(alllen[k])/alllen[k][-1])
    ax.plot(pts[:,0], pts[:,1],'.')
    #print alllen[k][1:10]

    #plt.figure()
    ###plt.plot(np.fft.fftfreq(len(allpars[k]),d=1/RESOLUTION), allpars[k])
    ###plt.title(k)
    ###plt.figure()
    ##plt.plot(graph[k][:,0],graph[k][:,1])
    ##plt.title(k)
  ##ax.plot(graph[k])
##ax.plot(graph[1][:,0],graph[1][:,1])

##ax.plot(allpars[16][::5])
##ax.plot(allpars[1][::5])

plt.show()



  
  