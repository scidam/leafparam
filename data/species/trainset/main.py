

import pdb

from cachepy import Cache
from scipy import interpolate
from scipy import signal
from skimage import feature
from skimage import measure
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.morphology import  disk
from skimage.morphology import  white_tophat
from skimage.segmentation import slic
from skimage.transform import resize
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.qda import QDA

from files import dataset
import matplotlib.pyplot as plt
import numpy as np


# 1. image filtering... 
# 2. 
mycache = Cache('test.dat')

RESOLUTION = float(1000)


def is_closed(contour):
  return (contour[0]==contour[-1]).all()


def length(contour):
  return np.sum(np.linalg.norm(np.array(contour[1:])-np.array(contour[:-1]),axis=1))
  

def get_largest(contours):
    _l = [length(x) for x in contours]
    return contours[np.argmax(_l)]


def parametrize(contour, resolution=RESOLUTION):
    _cont = np.array(contour)
    tck,u=interpolate.splprep([_cont[:,0],_cont[:,1]],s=0)
    unew = np.arange(0,1.0+1.0/resolution,1.0/resolution)
    r = np.array(interpolate.splev(unew,tck)).transpose()
    _l = [length(r[:j]) for j in range(1,len(r)+1)]
    _l = np.array(_l)/np.max(_l)
    ttck = interpolate.splrep(_l, unew)
    regu = interpolate.splev(unew, ttck)
    dr = np.array(interpolate.splev(regu,tck,der=1)).transpose()
    lzero =  dr[:,0] <= 0.0
    _t = np.arctan(dr[:,1]/dr[:,0])
    _t[lzero] = np.pi + _t[lzero]
    return (_t.tolist(), unew.tolist(), r)


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





  
labels = np.array(dataset.keys(), dtype=np.int)


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



# parametrization main loop

@mycache
def loadmydata():
    alla, allb, allc, alll = [], [], [], []
    indd=1
    for k in labels:
      for j in dataset[k]:
        im = imread(j)
        im = leaf_image_preprocess(im)
        img_filt = extract_leaf_stem(im)
        contours = measure.find_contours(img_filt, 0.8)
        a,b,c=parametrize(get_largest(contours))
        alla.append(a)
        allb.append(b)
        allc.append(c)
        alll.append(k)
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         cwtmatr = signal.cwt(signal.decimate(a,4),signal.ricker, np.linspace(0.0001,1,200))
        #toplt=[]
        #for x in cwtmatr: 
            #if any(x[x>2]):
                #toplt.append(np.mean(x[x>2]))
            #else:
                #toplt.append(0)
        #ax.set_xlim([0,160])
        #ax.set_ylim([-3,7])
        print j 
    return alla, allb, allc, alll

alla,allb,allc,alll = loadmydata()

print 'All contours loaded:', len(allc)



# storage = []
# 
# for a, l in zip(alla, alll):
#     print np.shape(a),l
#     storage.append(signal.cwt(signal.decimate(a,4),signal.ricker, np.linspace(0.001,1,200)))


pca = PCA(n_components=2)
pca.fit(allc[0])
res=pca.transform(allc[0])
scale = max(res[:,0])- min(res[:,0])
pca1 = PCA(n_components=2)
pca1.fit(allc[1])
res1=pca.transform(allc[1])
scale1 = max(res1[:,0])- min(res1[:,0])

plt.plot((res[:,0]-np.mean(res[:,0]))/scale, (res[:,1]-np.mean(res[:,1]))/scale)
plt.plot((res1[:,0]-np.mean(res1[:,0]))/scale1,  (res1[:,1]-np.mean(res1[:,1]))/scale1)
plt.show()

print len(storage)


#print np.shape(alllab), np.shape(allpars)

#lda = LDA()

#print cross_validation.cross_val_score(lda, allpars, alllab, cv=4),len(graph)






#x = np.linspace(0,1,10000)
#y = x*1.5*np.pi


#ppts=getpoints(y, x)
#ax.plot(ppts[:,0], ppts[:,1], 'r')
#plt.axis('equal')
#plt.show()



#for k in [1]:
    ###ffts = np.fft.fft(graph[k])
    ###freq = np.fft.fftfreq(np.shape(graph[k])[-1])
    ###print np.max(freq), np.min(freq)
    ###ffts[freq>0.2]=0.0
    #y = graph[k]
    #x = np.array(alllen[k])
    #widths = np.linspace(0,0.999,2000)+0.0001
    
    ##pts = getpoints(y,x)
    ##ax.plot(pts[:,0],pts[:,1],'.')
    ###print alllen[k][1:10]

    ###plt.figure()
    #####plt.plot(np.fft.fftfreq(len(allpars[k]),d=1/RESOLUTION), allpars[k])
    #####plt.title(k)
    #####plt.figure()
    ####plt.plot(graph[k][:,0],graph[k][:,1])
    ####plt.title(k)
  ####ax.plot(graph[k])
####ax.plot(graph[1][:,0],graph[1][:,1])

####ax.plot(allpars[16][::5])
####ax.plot(allpars[1][::5])

#plt.show()



  
  