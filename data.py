# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/Dancing2Music/License.txt
import os  
import pickle
import numpy as np
import random
import torch.utils.data
from torchvision.datasets import ImageFolder
import utils
import tqdm

class PoseDataset(torch.utils.data.Dataset):
	def __init__(self, data_dir, tolerance=False, category='full'):
		self.data_dir = data_dir
		z_fname = '{}/unitList/zumba_unit.txt'.format(data_dir)
		b_fname = '{}/unitList/ballet_unit.txt'.format(data_dir)
		h_fname = '{}/unitList/hiphop_unit.txt'.format(data_dir)
		self.z_data = []
		self.b_data = []
		self.h_data = []
		with open(z_fname, 'r') as f:
			for line in f:
				self.z_data.append([s for s in line.strip().split(' ')])
		with open(b_fname, 'r') as f:
			for line in f:
				self.b_data.append([s for s in line.strip().split(' ')])
		with open(h_fname, 'r') as f:
			for line in f:
				self.h_data.append([s for s in line.strip().split(' ')])
		
		self.cate = category
		if self.cate == 'ballet':
			self.data = [self.b_data]
		else:
			self.data = [self.z_data, self.b_data, self.h_data]

		self.tolerance = tolerance
		if self.tolerance:
			z3_fname = '{}/unitList/zumba_unitseq3.txt'.format(data_dir)
			b3_fname = '{}/unitList/ballet_unitseq3.txt'.format(data_dir)
			h3_fname = '{}/unitList/hiphop_unitseq3.txt'.format(data_dir)
			z4_fname = '{}/unitList/zumba_unitseq4.txt'.format(data_dir)
			b4_fname = '{}/unitList/ballet_unitseq4.txt'.format(data_dir)
			h4_fname = '{}/unitList/hiphop_unitseq4.txt'.format(data_dir)
			z3_data = []; b3_data = []; h3_data = []; z4_data = []; b4_data = []; h4_data = []
			with open(z3_fname, 'r') as f:
				for line in f:
					z3_data.append([s for s in line.strip().split(' ')])
			with open(b3_fname, 'r') as f:
				for line in f:
					b3_data.append([s for s in line.strip().split(' ')])
			with open(h3_fname, 'r') as f:
				for line in f:
					h3_data.append([s for s in line.strip().split(' ')])
			with open(z4_fname, 'r') as f:
				for line in f:
					z4_data.append([s for s in line.strip().split(' ')])
			with open(b4_fname, 'r') as f:
				for line in f:
					b4_data.append([s for s in line.strip().split(' ')])
			with open(h4_fname, 'r') as f:
				for line in f:
					h4_data.append([s for s in line.strip().split(' ')])
			self.zt_data = z3_data + z4_data
			self.bt_data = b3_data + b4_data
			self.ht_data = h3_data + h4_data
			
			if self.cate == 'ballet':
				self.t_data = [self.bt_data]
			else:
				self.t_data = [self.zt_data, self.bt_data, self.ht_data]

		self.mean_pose=np.load(data_dir+'/stats/all_onbeat_mean.npy')
		self.std_pose=np.load(data_dir+'/stats/all_onbeat_std.npy')

	def __getitem__(self, index):
		cls = 0
		if self.cate == 'full':
			cls = random.randint(0,2)
		
		if self.tolerance and random.randint(0,9)==0:
			index = random.randint(0, len(self.t_data[cls])-1)
			path = self.t_data[cls][index][0]
			path = os.path.join(self.data_dir, path[5:])
			orig_poses = np.load(path)
			sel = random.randint(0, orig_poses.shape[0]-1)
			orig_poses = orig_poses[sel]
		else:
			index = random.randint(0, len(self.data[cls])-1)
			path = self.data[cls][index][0]
			path = os.path.join(self.data_dir, path[5:])
			orig_poses = np.load(path)

		xjit = np.random.uniform(low=-50, high=50)
		yjit = np.random.uniform(low=-20, high=20)
		poses = orig_poses.copy()
		poses[:,:,0] += xjit
		poses[:,:,1] += yjit
		xjit = np.random.uniform(low=-50, high=50)
		yjit = np.random.uniform(low=-20, high=20)
		poses2 = orig_poses.copy()
		poses2[:,:,0] += xjit
		poses2[:,:,1] += yjit

		poses = poses.reshape(poses.shape[0], poses.shape[1]*poses.shape[2])
		poses2 = poses2.reshape(poses2.shape[0], poses2.shape[1]*poses2.shape[2])
		for i in range(poses.shape[0]):
			poses[i] = (poses[i]-self.mean_pose)/self.std_pose
			poses2[i] = (poses2[i]-self.mean_pose)/self.std_pose

		return torch.Tensor(poses), torch.Tensor(poses2)

	def __len__(self):
		if self.cate == 'ballet':
			return len(self.b_data)
		else:
			return len(self.z_data)+len(self.b_data) + len(self.h_data)


class MovementAudDataset(torch.utils.data.Dataset):
	def __init__(self, data_dir, category='full'):
		self.data_dir = data_dir
		z3_fname = '{}/unitList/zumba_unitseq3.txt'.format(data_dir)
		b3_fname = '{}/unitList/ballet_unitseq3.txt'.format(data_dir)
		h3_fname = '{}/unitList/hiphop_unitseq3.txt'.format(data_dir)
		z4_fname = '{}/unitList/zumba_unitseq4.txt'.format(data_dir)
		b4_fname = '{}/unitList/ballet_unitseq4.txt'.format(data_dir)
		h4_fname = '{}/unitList/hiphop_unitseq4.txt'.format(data_dir)
		self.z3_data = []
		self.b3_data = []
		self.h3_data = []
		self.z4_data = []
		self.b4_data = []
		self.h4_data = []
		with open(z3_fname, 'r') as f:
			for line in f:
				self.z3_data.append([s for s in line.strip().split(' ')])
		with open(b3_fname, 'r') as f:
			for line in f:
				self.b3_data.append([s for s in line.strip().split(' ')])
		with open(h3_fname, 'r') as f:
			for line in f:
				self.h3_data.append([s for s in line.strip().split(' ')])
		with open(z4_fname, 'r') as f:
			for line in f:
				self.z4_data.append([s for s in line.strip().split(' ')])
		with open(b4_fname, 'r') as f:
			for line in f:
				self.b4_data.append([s for s in line.strip().split(' ')])
		with open(h4_fname, 'r') as f:
			for line in f:
				self.h4_data.append([s for s in line.strip().split(' ')])
		# MODIFIED
		self.cate = category
		z_data_root = 'zumba/'
		b_data_root = 'ballet/'
		h_data_root = 'hiphop/'
		if category == 'full':
			self.data_3 = [self.z3_data, self.b3_data, self.h3_data]
			self.data_4 = [self.z4_data, self.b4_data, self.h4_data]
			self.data_root = [z_data_root, b_data_root, h_data_root ]
		elif category == 'zumba':
			self.data_3 = [self.z3_data]
			self.data_4 = [self.z4_data]
			self.data_root = [z_data_root]
		elif category == 'ballet':
			self.data_3 = [self.b3_data]
			self.data_4 = [self.b4_data]
			self.data_root = [b_data_root]
		elif category == 'hiphop':
			self.data_3 = [self.h3_data]
			self.data_4 = [self.h4_data]
			self.data_root = [h_data_root]

		self.mean_pose=np.load(data_dir+'/stats/all_onbeat_mean.npy')
		self.std_pose=np.load(data_dir+'/stats/all_onbeat_std.npy')
		self.mean_aud=np.load(data_dir+'/stats/all_aud_mean.npy')
		self.std_aud=np.load(data_dir+'/stats/all_aud_std.npy')

	def __getitem__(self, index):

		if self.cate == 'full':
			cls = random.randint(0,2)
			isthree = random.randint(0,1)

			if isthree == 0:
				index = random.randint(0, len(self.data_4[cls])-1)
				path = self.data_4[cls][index][0]
			else:
				index = random.randint(0, len(self.data_3[cls])-1)
				path = self.data_3[cls][index][0]
		elif self.cate == 'ballet':
			cls = 0
			isthree = 1
			'''
			if index >= len(self.b3_data):
				index = index - len(self.b3_data)
				path = self.data_4[0][index][0]
			else:
				path = self.data_3[0][index][0]
			'''
			path = self.data_4[0][index][0]
		elif self.cate == 'zumba':
			cls = 0
			isthree = 1
			'''
			if index >= len(self.z3_data):
				index = index - len(self.z3_data)
				path = self.data_4[0][index][0]
			else:
				path = self.data_3[0][index][0]
			'''
			path = self.data_4[0][index][0]
		elif self.cate == 'hiphop':
			cls = 0
			isthree = 1
			'''
			if index >= len(self.h3_data):
				index = index - len(self.h3_data)
				path = self.data_4[0][index][0]
			else:
				path = self.data_3[0][index][0]
			'''
			path = self.data_4[0][index][0]

		path = os.path.join(self.data_dir, path[5:])
		stdpSeq = np.load(path)
		vid, cid = path.split('/')[-4], path.split('/')[-3]
		#vid, cid = vid_cid[:11], vid_cid[12:]
		aud = np.load('{}/{}/{}/{}/aud/c{}_fps15.npy'.format(self.data_dir, self.data_root[cls], vid, cid, cid))

		stdpSeq = stdpSeq.reshape(stdpSeq.shape[0], stdpSeq.shape[1], stdpSeq.shape[2]*stdpSeq.shape[3])
		for i in range(stdpSeq.shape[0]):
			for j in range(stdpSeq.shape[1]):
				stdpSeq[i,j] = (stdpSeq[i,j]-self.mean_pose)/self.std_pose
		if isthree == 0:
			start = random.randint(0,1)
			stdpSeq = stdpSeq[start:start+3]

		for i in range(aud.shape[0]):
			aud[i] = (aud[i]-self.mean_aud)/self.std_aud
		aud = aud[:30]
		return torch.Tensor(stdpSeq), torch.Tensor(aud)

	def __len__(self):
		if self.cate == 'full':
			return len(self.z3_data)+len(self.b3_data)+len(self.z4_data)+len(self.b4_data)+len(self.h3_data)+len(self.h4_data)
		elif self.cate == 'zumba':
			return len(self.z4_data) #+ len(self.z4_data)
		elif self.cate == 'ballet':
			return len(self.b4_data) #+ len(self.b4_data)
		elif self.cate == 'hiphop':
			return len(self.h4_data) #+ len(self.h4_data)

class AudDataset(torch.utils.data.Dataset):
	def __init__(self, data_dir):
		self.data_dir = data_dir
		z3_fname = '{}/unitList/zumba_unitseq3.txt'.format(data_dir)
		b3_fname = '{}/unitList/ballet_unitseq3.txt'.format(data_dir)
		h3_fname = '{}/unitList/hiphop_unitseq3.txt'.format(data_dir)
		z4_fname = '{}/unitList/zumba_unitseq4.txt'.format(data_dir)
		b4_fname = '{}/unitList/ballet_unitseq4.txt'.format(data_dir)
		h4_fname = '{}/unitList/hiphop_unitseq4.txt'.format(data_dir)
		self.z3_data = []
		self.b3_data = []
		self.h3_data = []
		self.z4_data = []
		self.b4_data = []
		self.h4_data = []
		with open(z3_fname, 'r') as f:
			for line in f:
				self.z3_data.append([s for s in line.strip().split(' ')])
		with open(b3_fname, 'r') as f:
			for line in f:
				self.b3_data.append([s for s in line.strip().split(' ')])
		with open(h3_fname, 'r') as f:
			for line in f:
				self.h3_data.append([s for s in line.strip().split(' ')])
		with open(z4_fname, 'r') as f:
			for line in f:
				self.z4_data.append([s for s in line.strip().split(' ')])
		with open(b4_fname, 'r') as f:
			for line in f:
				self.b4_data.append([s for s in line.strip().split(' ')])
		with open(h4_fname, 'r') as f:
			for line in f:
				self.h4_data.append([s for s in line.strip().split(' ')])
		
		z_data_root = 'zumba/'
		b_data_root = 'ballet/'
		h_data_root = 'hiphop/'

		self.data_3 = self.z3_data + self.b3_data + self.h3_data
		self.data_4 = self.z4_data + self.b4_data + self.h4_data
		
		self.data_len = [len(self.z3_data), 
			len(self.z3_data) + len(self.b3_data), 
			len(self.z3_data) + len(self.b3_data) + len(self.h3_data),
			len(self.z3_data) + len(self.b3_data) + len(self.h3_data) + len(self.z4_data), 
			len(self.z3_data) + len(self.b3_data) + len(self.h3_data) + len(self.z4_data) + len(self.b4_data)]

		self.data_root = [z_data_root, b_data_root, h_data_root]

		self.mean_aud=np.load(data_dir+'/stats/all_aud_mean.npy')
		self.std_aud=np.load(data_dir+'/stats/all_aud_std.npy')

	def __len__(self):
		return len(self.data_3) + len(self.data_4)

	def __getitem__(self, index):

		if index >= len(self.data_3):
			path = self.data_4[index-len(self.data_3)][0]
		else:
			path = self.data_3[index][0]

		cls = -1
		for i in range(6):
			if (index < self.data_len[0]) or \
				(index >= self.data_len[2] and index < self.data_len[3]): 
				cls = 0
			elif (index >= self.data_len[0] and index < self.data_len[1]) or \
				(index >= self.data_len[3] and index < self.data_len[4]):
				cls = 1
			else:
				cls = 2
		#print('path', path)
		path = os.path.join(self.data_dir, path[5:])
		vid, cid = path.split('/')[-4], path.split('/')[-3]
		aud = np.load('{}/{}/{}/{}/aud/c{}_fps15.npy'.format(self.data_dir, self.data_root[cls], vid, cid, cid))

		for i in range(aud.shape[0]):
			aud[i] = (aud[i]-self.mean_aud)/self.std_aud
		#print('aud', aud.shape)
		aud = aud[:59]
		
		return (torch.Tensor(aud), cls)

def get_loader(batch_size, shuffle, num_workers, dataset, data_dir, tolerance=False, eval='full', random_seed=None):
	if dataset == 0:
		a2d = PoseDataset(data_dir, tolerance, category=eval)
	elif dataset == 1:
		if random_seed != None:
			a2d = AudDataset(data_dir)
			data_size = len(a2d)
			indices = list(range(data_size))
			split = int(np.floor(0.2 * data_size))
			np.random.seed(random_seed)
			np.random.shuffle(indices)
			train_idx, val_idx = indices[split:], indices[:split]
			train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
			val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
			train_loader = torch.utils.data.DataLoader(dataset=a2d,
														batch_size=batch_size,
														sampler=train_sampler,
														num_workers=num_workers,)
			val_loader = torch.utils.data.DataLoader(dataset=a2d,
														batch_size=batch_size,
														sampler=val_sampler,
														num_workers=num_workers,)
			return train_loader, val_loader
		else:
			a2d = AudDataset(data_dir)
	elif dataset == 2:
		if eval == 'full':
			a2d = MovementAudDataset(data_dir)
		else:
			a2d = MovementAudDataset(data_dir, category=eval)

	data_loader = torch.utils.data.DataLoader(dataset=a2d,
											batch_size=batch_size,
											shuffle=shuffle,
											num_workers=num_workers,
											)
	return data_loader


if __name__ == "__main__":
	
	data_dir = "/home/weiyu/dance/Dancing2Music/data"
	data_loader = get_loader(1, True, 4, 1, data_dir)
	min_len = 10000
	for i, (aud, lab) in tqdm.tqdm(enumerate(data_loader)):
		if aud.shape[1] < min_len:
			min_len = aud.shape[1]
	print(min_len)
