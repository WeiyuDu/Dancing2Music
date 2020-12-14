# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/Dancing2Music/License.txt
import os
import time
import numpy as np
import random
import math

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from utils import Logger
import argparse
from networks import MusicStyleExtractor
from data import get_loader

class Trainer_StyleExtractor(object):
	def __init__(self, train_loader, val_loader, model, args=None):
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.model = model

		self.args = args
		self.logger = Logger(args.log_dir)
		self.logs = self.init_logs()
		self.log_interval = args.log_interval
		self.snapshot_ep = args.snapshot_ep
		self.snapshot_dir = args.snapshot_dir

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

		self.cross_entropy = torch.nn.CrossEntropyLoss()
		
		self.best_accuracy = 0
		self.best_epoch = -1

	def init_logs(self):
		return {'cross_entropy':0}

	def forward(self, aud):
		return self.model(aud)

	def update(self, pred, gt):
		self.optimizer.zero_grad()
		self.loss = self.cross_entropy(pred, gt)
		self.loss.backward()
		self.optimizer.step()

	def save(self, filename, ep, total_it):
		state = {
				'model': self.model.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'ep': ep,
				'total_it': total_it
				}
		torch.save(state, filename)
		return

	def resume(self, model_dir, train=True):
		checkpoint = torch.load(model_dir)
		# weight
		self.model.load_state_dict(checkpoint['model'])
		# optimizer
		if train:
			self.optimizer.load_state_dict(checkpoint['optimizer'])
		return checkpoint['ep'], checkpoint['total_it']

	def cuda(self):
		self.model.to(self.args.device)

	def train(self, ep=0, it=0):
		self.cuda()

		for epoch in range(ep, self.args.num_epochs):
			self.model.train()
			correct_acc = 0
			total_acc = 0

			for i, (aud, gt) in enumerate(self.train_loader):
				aud = aud.to(self.args.device)
				gt = gt.to(self.args.device)
				pred = self.forward(aud)
				self.update(pred, gt)

				pred_lab = torch.argmax(pred, axis=1)
				correct_num = (pred_lab == gt).sum()
				correct_acc += correct_num 
				total_acc += aud.shape[0]
				
				self.logs['cross_entropy'] += self.loss.data

				it += 1
				if it % self.log_interval == 0:
					#print('Epoch:{:3} Iter{}/{}\loss {:.3f}'.format(
					#epoch, i, len(self.train_loader), self.loss))

					for tag, value in self.logs.items():
						self.logger.scalar_summary(tag, value/self.log_interval, it)
					self.logs = self.init_logs()

			train_accuracy = correct_acc / total_acc
			print("train accuracy at epoch ", peoch, " is ", train_accuracy)
			val(epoch)

			if epoch % self.snapshot_ep == 0:
				self.save(os.path.join(self.snapshot_dir, '{:04}.ckpt'.format(epoch)), epoch, it)

	def val(self, epoch):
		self.cuda()
		self.model.eval()
			
		with torch.no_grad():
			correct_acc = 0
			total_acc = 0
			for i, (aud, gt) in enumerate(self.val_loader):
				aud = aud.to(self.args.device)
				gt = gt.to(self.args.device)
				pred = self.forward(aud)
				pred_lab = torch.argmax(pred, axis=1)
				correct_num = (pred_lab == gt).sum()
				correct_acc += correct_num.item()
				total_acc += aud.shape[0]
				
			accuracy = correct_acc / total_acc
			print("accuracy at epoch ", epoch, "is ", accuracy)
			if (accuracy > self.best_accuracy):
				self.best_accuracy = accuracy
				self.best_epoch = epoch
				if epoch % self.snapshot_ep != 0:
					self.save(os.path.join(self.snapshot_dir, '{:04}.ckpt'.format(epoch)), epoch, it)
				print("saved best model saved at epoch ", epoch)

class StyleOptions():
	def __init__(self):
		parser = argparse.ArgumentParser()
		
		# custom starts
		parser.add_argument('--train', type=bool, default=False)
		parser.add_argument('--val_epoch', type=int, default=100)
		parser.add_argument('--model_dir', type=str, default="snapshot/Style/0100.ckpt")
		parser.add_argument('--device', type=str, default="cuda:3")
		parser.add_argument('--name', default=None)
		parser.add_argument('--random_seed', type=int, default=1)

		parser.add_argument('--log_interval', type=int, default=50)
		parser.add_argument('--log_dir', default='./logs')
		parser.add_argument('--snapshot_ep', type=int, default=20)
		parser.add_argument('--snapshot_dir', default='./snapshot')
		parser.add_argument('--data_dir', default='/home/weiyu/dance/Dancing2Music/data/')

		# Model architecture
		parser.add_argument('--pose_size', type=int, default=28)
		parser.add_argument('--dim_z_motion', type=int, default=10)
		parser.add_argument('--hidden_size', type=int, default=512)
		parser.add_argument('--output_size', type=int, default=30)

		# Training
		parser.add_argument('--lr', type=float, default=2e-4)
		parser.add_argument('--batch_size', type=int, default=256)
		parser.add_argument('--num_epochs', type=int, default=200)
		parser.add_argument('--num_layers', type=int, default=2)
		parser.add_argument('--num_cls', type=int, default=3)
		parser.add_argument('--shuffle', type=bool, default=True)

		# Others
		parser.add_argument('--num_workers', type=int,  default=4)
		parser.add_argument('--resume', default=None)
		parser.add_argument('--dataset', type=int, default=1)
		parser.add_argument('--tolerance', action='store_true')
		parser.add_argument('--category', default='full')
		self.parser = parser

	def parse(self):
		self.opt = self.parser.parse_args()
		args = vars(self.opt)
		return self.opt

if __name__ == "__main__":
	parser = StyleOptions()
	args = parser.parse()

	args.log_dir = os.path.join(args.log_dir, args.name)
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)
	args.snapshot_dir = os.path.join(args.snapshot_dir, args.name)
	if not os.path.exists(args.snapshot_dir):
		os.mkdir(args.snapshot_dir)

	model = MusicStyleExtractor(args.dim_z_motion, 
		hidden_size=args.hidden_size, 
		output_size=args.output_size, 
		pose_size=args.pose_size, 
		cls=args.num_cls, 
		num_layers=args.num_layers)
	
	train_loader, val_loader = get_loader(args.batch_size, args.shuffle, 
											args.num_workers, args.dataset, 
											args.data_dir, tolerance=False, 
											eval='full', random_seed=args.random_seed)

	trainer = Trainer_StyleExtractor(train_loader, val_loader, model, args=args)

	if not args.resume is None:
		ep, it = trainer.resume(args.resume, True)
	else:
		ep, it = 0, 0

	if args.train:
		trainer.train(ep, it)
	else:
		_, _ = trainer.resume(args.model_dir, train=False)
		data_loader = get_loader(args.batch_size, args.shuffle, 
									args.num_workers, args.dataset, 
									args.data_dir, tolerance=False, 
									eval='full', random_seed=None)
		trainer.val_loader = data_loader
		trainer.val(args.val_epoch)