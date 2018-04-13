#!/usr/bin/python
#
import warnings
warnings.filterwarnings('ignore')

from time import time
import numpy as np
import h5py, glob, os, argparse
from sys import stdout 

from sklearn.svm.classes import SVC
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, metrics)
from sklearn.cluster import KMeans
import matplotlib as mplot
from matplotlib import pyplot as plt


def plot_embedding(ranges, zones, probs, xx, yy, reduced_data, beats, centroids, label, audioname):
  x_min, x_max, y_min, y_max = ranges

  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(1,1,1,axisbg=(0.95, 0.95, 0.95))
  fig.subplots_adjust(left=0.05, right=0.98,  bottom=0.05, top=0.98, wspace=0.05) #, wspace=None, hspace=None
  #left=-2, bottom=-2, right=2, top=2
  ax.set_xlim([x_min, x_max])
  ax.set_ylim([y_min, y_max])
  #plt.xticks([]), plt.yticks([])
  ax.grid(color='w', linestyle='--')
  for child in ax.get_children():
    if isinstance(child, mplot.spines.Spine):
        child.set_color('w')
  #ax.imshow(zones, interpolation='hamming', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
  #           cmap=plt.cm.Pastel1,
  #           aspect='auto', origin='lower')

  #if not probs is None: 
  #  ax.imshow(probs, interpolation='hamming', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
  #             cmap=plt.cm.binary,
  #             aspect='auto', origin='lower', alpha=0.4)

  if label=='trained':
    large = np.where(beats>=reduced_data.shape[0])[0]
    if len(large) > 1:
      beats = beats[:large[0]]
    pos_beats=reduced_data[beats,:]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], color=plt.cm.Set1(1))
    ax.scatter(pos_beats[:, 0], pos_beats[:, 1], s=16, color='r')
  else:
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], color=plt.cm.Set1(1))
  
  #ax.scatter(centroids[:, 0], centroids[:, 1],
  #            marker='x', s=169, linewidths=3,
  #            color='k', zorder=10)
  fig_name=('{}/images/{}_{}.png'.format(exp_folder, label, audioname))
  fig.savefig(fig_name)
  #exit()

def make_video(ranges, zones, probs, xx, yy, reduced_data, beats, label, audioname):
  x_min, x_max, y_min, y_max = ranges
  vid_name=('{}/images/{}_{}.mp4'.format(exp_folder, label, audioname))
  if label=='trained':
    large = np.where(beats>=reduced_data.shape[0])[0]
    if len(large) > 1:
      beats = beats[:large[0]]
    colors = np.ones((reduced_data.shape[0]), dtype=np.int)*1
    shape = np.ones((reduced_data.shape[0]))*2
    colors[beats]=0
    shape[beats]=15

  fig = plt.figure(figsize=(8,6))
  ax = fig.add_subplot(1,1,1,axisbg=(0.95, 0.95, 0.95))
  fig.subplots_adjust(left=0.05, right=0.98,  bottom=0.05, top=0.98, wspace=0.05)
  #fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

  ax.set_xlim([x_min, x_max])
  ax.set_ylim([y_min, y_max])
  ax.grid(color='w', linestyle='--')
  for child in ax.get_children():
    if isinstance(child, mplot.spines.Spine):
        child.set_color('w')
  #plt.xticks([]), plt.yticks([])

  #ax.imshow(zones, interpolation='hamming',
  #           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
  #           cmap=plt.cm.Pastel1,
  #           aspect='auto', origin='lower')

  #if not probs is None: 
  #  ax.imshow(probs, interpolation='hamming',
  #             extent=(xx.min(), xx.max(), yy.min(), yy.max()),
  #             cmap=plt.cm.binary,
  #             aspect='auto', origin='lower', alpha=0.4)

  stop_frame=3600 if reduced_data.shape[0] >= 3600 else reduced_data.shape[0]
  for i in range(stop_frame):

    if label=='trained':
      if i>0:
        nc=np.copy(colors[:i])
        nc[0]=9
        nc[-1]=1
        ns=np.copy(shape[:i])
        ns[-1]=35
      else:
        nc = plt.cm.Set1(0)
        ns = 5
      indata = ax.scatter(reduced_data[:i, 0], reduced_data[:i, 1], 
              marker='.', s=ns, c=nc, cmap=plt.cm.Set1)
    else:
      start=0 if i < 300 else i-300
      length=i if i < 300 else 300
      nc = np.ones((length))
      ns = np.ones((length))*2
      if i>0:
          nc[-1]=0
          ns[-1]=35
      else:
          nc = plt.cm.Set1(0)
          ns = 5
      indata = ax.scatter(reduced_data[start:i, 0], reduced_data[start:i, 1], 
              marker='x', s=ns, c=nc, cmap=plt.cm.Set1)

    fig.savefig('{}/{}_{}.png'.format(temp_folder, label, i))
    #plt.close()
    indata.remove()
    stdout.write('current frame:{}\r'.format(i))
    stdout.flush()

  cmmd='ffmpeg -y -start_number 0 -r 30 -i {}/{}_%d.png -vb 5M -c:v mpeg4 {}'.format(temp_folder, label, vid_name)
  os.system(cmmd)
  for t in glob.glob('{}/{}*.png'.format(temp_folder, label)):
    os.remove(t)

def main():
  Cleanfiles = sorted(glob.glob('{}/evaluation/end2end_*_Clean_feats.h5'.format(exp_folder)))
  Untrainedfiles = sorted(glob.glob('{}/untrained/*.h5'.format(exp_folder)))
  data= dict()
  for i in range(len(Cleanfiles)):
    data[str(i)] = dict()
    data[str(i)]['file'] = Cleanfiles[i]
    audioname = os.path.basename(Cleanfiles[i])
    audioname = audioname.replace('end2end_', '')
    audioname = audioname.replace('test_', '')
    audioname = audioname.replace('{}_'.format(args.exp), '')
    audioname = audioname.replace('_Clean_feats.h5', '')
    data[str(i)]['audioname'] = audioname
    befr = np.unique(np.loadtxt('{}/Annotations/corrected/{}.txt'.format(args.data, audioname)))
    befr = np.asarray(befr*30, dtype=np.int)
    data[str(i)]['beats']= befr
    
    with h5py.File(Cleanfiles[i]) as f:
      feats = np.zeros(f['feats'].shape)
      np.copyto(feats, f['feats'])
    data[str(i)]['feats']= feats

    #Music Change for beat step
    if k_zones<=4:
      target_steps=np.zeros((feats.shape[0]), dtype=np.int)
      start_step = int(np.average((befr[0], befr[1])))
      for j in range(1, befr.shape[0]-1):
          change_step=int(np.average((befr[j], befr[j+1])))
          target_steps[start_step:change_step]=((j-1)%4)+1
          start_step = change_step 

      data[str(i)]['steps']= target_steps

  if len(data)>1:
    length = data['0']['feats'].shape[0]
    amax = 3600 if  len(data) > 3 else int(length/2) 
    feats = data['0']['feats'][:amax]
    if k_zones<=4:
      target_steps = data['0']['steps'][:amax]
    for i in range(1, len(data)):
      length = data[str(i)]['feats'].shape[0]
      amax = 3600 if  len(data) > 3 else int(length/2) 
      feats = np.concatenate((feats, data[str(i)]['feats'][:amax]), axis=0)
      if k_zones<=4:
        target_steps = np.concatenate((target_steps, data[str(i)]['steps'][:amax]), axis=0)
  else:
    feats = data['0']['feats']
    if k_zones<=4:
      target_steps = data['0']['steps']

  embedder = decomposition.PCA(n_components=2, svd_solver='randomized')
  reduced_data = embedder.fit_transform(feats)
  kmeans = KMeans(init='k-means++', n_clusters=k_zones, n_init=10)
  kmeans.fit(reduced_data)
  centroids = kmeans.cluster_centers_
  # Step size of the mesh. Decrease to increase the quality of the VQ.
  h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

  # Plot the decision boundary. For that, we will assign a color to each
  x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
  y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # Obtain labels for each point in mesh. Use last trained model.
  Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)

  if k_zones<=4:
    bg2 = SVC(probability=True, C=1, gamma=1).fit(reduced_data, target_steps)
    Z1 = bg2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 0]
    Z1 = Z1.reshape(xx.shape)
  else:
    Z1 = None

  for i in range(len(Cleanfiles)):
    reduced=embedder.transform(data[str(i)]['feats'])
    beats = data[str(i)]['beats']
    audioname = data[str(i)]['audioname']
    plot_embedding([x_min, x_max, y_min, y_max], Z, Z1, xx, yy, reduced,\
                   beats, centroids, 'trained', audioname)
    make_video([x_min, x_max, y_min, y_max], Z, Z1, xx, yy, reduced,\
                   beats, 'trained', audioname)

  for i in range(len(Untrainedfiles)):
    with h5py.File(Untrainedfiles[i]) as f:
      ufeats = np.zeros(f['feats'].shape)
      np.copyto(ufeats, f['feats'])
    ureduced=embedder.transform(ufeats)
    audioname = os.path.basename(Untrainedfiles[i]).split('.')[0]
    plot_embedding([x_min, x_max, y_min, y_max], Z, Z1, xx, yy, ureduced,\
                   None, centroids, 'untrained', audioname)
    make_video([x_min, x_max, y_min, y_max], Z, Z1, xx, yy, ureduced,\
                   None, 'untrained', audioname)

  try:
    os.rmdir(temp_folder)
  except Exception as e:
    pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='BPM Evaluation')
  parser.add_argument('--exp', '-e', type=str, help='Experiment type')
  parser.add_argument('--encoder', '-c', type=str, help='Audio Encoder')
  parser.add_argument('--net', '-n', type=str, help='Network model')
  parser.add_argument('--data', type=str, help='Data Folder')
  parser.add_argument('--initstp', '-i', type=int, help='Initial step', default=0)
  parser.add_argument('--k_zones','-k', type=int, help='K-means zones', default=2)
  args = parser.parse_args()
  k_zones = args.k_zones
  exp_folder='./exp/{}/{}_quat_{}_initstep_{}'.format(args.exp, args.net, args.encoder, args.initstp)
  temp_folder='{}/temp'.format(exp_folder)
  if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
  main()