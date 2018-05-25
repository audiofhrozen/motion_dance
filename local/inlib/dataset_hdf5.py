from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer
from chainer import dataset
from chainer import training
import h5py
import glob
import logging
import numpy as np


class DanceSeqHDF5(dataset.DatasetMixin):
  def __init__(self, folder, sequence, stage, init_step):
    self._inputs = ["input", "current"] 
    self.sequence = sequence
    self.steps = 1
    logging.info('Searching in {}/{} for files:'.format(folder, stage))
    self.list_file = glob.glob('{}/{}_motion*'.format(folder, stage))
    index = []
    for i in range(len(self.list_file)):
      with h5py.File(self.list_file[i], 'r') as f:
        current_lenght=f[self._inputs[0]].shape[0]
        if self.sequence >= current_lenght:
          logging.error('The lenght of the sequence is larger thant the lenght of the file...')
          raise ValueError('') 
        max_size =  current_lenght - (self.sequence + self.steps ) 
        if not '_dims' in locals():
          _dims = [ None ] * len(self._inputs)
          _types = [ None ] * len(self._inputs)
          for j in range(len(self._inputs)):
            testfile= f[self._inputs[j]]
            _dim = testfile[0].shape
            _dims[j] = _dim if len(_dim) !=0 else []
            _types[j] = testfile.dtype
            logging.info('  data label: {} \t dim: {} \t dtype: {}'.format(self._inputs[j], list(_dims[j]), _types[i]))                    
      _index = [[i, x] for x in np.arange(max_size)]
      index +=_index
    try:
      self._dims = _dims
    except Exception as e:
      logging.error('Cannot assign dimensions, data not found...')
      raise TypeError(e)
    self._type = _types
    self.idxs = index
    self.init_step = init_step
    logging.info('sequence: {}'.format(self.sequence))
    logging.info('Total of {} files...'.format(len(self.idxs)))
      

  def __len__(self):
    return len(self.idxs)

  def get_example(self, i):
    iDB, iFL = self.idxs[i]
    data_labels = [None]*3
    with h5py.File(self.list_file[iDB], 'r') as f:
      data_labels[0] = f[self._inputs[0]][iFL: iFL + self.sequence][None,:]
      if self.init_step == 0:
        data_labels[1] = np.zeros((1,71), dtype=np.float32) #TODO: to variable size
      else:
        data_labels[1] = f[self._inputs[1]][iFL: iFL + self.steps]
      data_labels[2] = f[self._inputs[1]][iFL + self.steps: iFL + self.steps+ self.sequence][None,:]  
    return data_labels
