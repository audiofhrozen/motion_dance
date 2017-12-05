import numpy as np

class Configurations(object):
	def __init__(self, _step, _data, _exp, _rot):
		self.next_step = _step
		self.data_folder = _data 
		self.exp = _exp
		self.rot_type= _rot # euler, quat

		self.wlen = 160	
		self.hop = 80
		self.fps = 30
		self.frq_smp = 16000
		self.rng_pos = [-0.9, 0.9]
		self.rng_wav = [-0.9, 0.9]

		self.add_silence = 60 #in secs

		self.indexes = np.linspace(0,self.frq_smp, num=31, dtype=np.int)
		self.parts = ['pelvis', 'pelvis', 'head', 'neck_01', 'spine_01', 'spine_02', 'upperarm_l', 'upperarm_r', 'lowerarm_l', 
		    'lowerarm_r', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r', 'clavicle_l', 'clavicle_r']
		
		self.out_folder = None
		self.file_pos_minmax = None 
		self.correct_file = None
		self.correct_name = None
		self.correct_frm = None
		self.snr_lst =None

		frame_lenght = 0
		for fps in range(self.fps):
			frame_lenght = np.amax((frame_lenght, self.indexes[fps+1] - self.indexes[fps]))

		self.frame_lenght=frame_lenght
		self.intersec_pos= None
		self.slope_pos= None
		self.intersec_wav= None
		self.slope_wav= None
		self.pos_scale = 100.0
		
		if self.rot_type=='euler':
			self.pos_dim= len(self.parts)*3
		elif self.rot_type=='quat':
			self.pos_dim=len(self.parts)*4-1
		else:
			raise TypeError('Incorrect type of rotation')
		return
		pass

		
