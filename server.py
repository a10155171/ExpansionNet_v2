"""
	Small script for testing on few generic images given the model weights.
	In order to minimize the requirements, it runs only on CPU and images are
	processed one by one.
"""

import socket
import threading

import torch
import argparse
import pickle
from argparse import Namespace

from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description


class model_server :
	def __init__(self, parser):
		self.__args = parser.parse_args()

		drop_args = Namespace(enc=0.0,
							dec=0.0,
							enc_input=0.0,
							dec_input=0.0,
							other=0.0)
		model_args = Namespace(model_dim=self.__args.model_dim,
							N_enc=self.__args.N_enc,
							N_dec=self.__args.N_dec,
							drop_args=drop_args)

		with open('./ExpansionNet_v2/demo_material/demo_coco_tokens.pickle', 'rb') as f:
			self.__coco_tokens = pickle.load(f)
			self.__sos_idx = self.__coco_tokens['word2idx_dict'][self.__coco_tokens['sos_str']]
			self.__eos_idx = self.__coco_tokens['word2idx_dict'][self.__coco_tokens['eos_str']]

		print("Dictionary loaded ...")

		self.__img_size = 384
		self.__model = End_ExpansionNet_v2(swin_img_size=self.__img_size, swin_patch_size=4, swin_in_chans=3,
									swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
									swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
									swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
									swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
									swin_use_checkpoint=False,
									final_swin_dim=1536,

									d_model=model_args.model_dim, N_enc=model_args.N_enc,
									N_dec=model_args.N_dec, num_heads=8, ff=2048,
									num_exp_enc_list=[32, 64, 128, 256, 512],
									num_exp_dec=16,
									output_word2idx=self.__coco_tokens['word2idx_dict'],
									output_idx2word=self.__coco_tokens['idx2word_list'],
									max_seq_len=self.__args.max_seq_len, drop_args=model_args.drop_args,
									rank=self.__args.device)
		checkpoint = torch.load(self.__args.load_path, map_location=torch.device(self.__args.device))
		self.__model.load_state_dict(checkpoint['model_state_dict'])
		self.__model.to(self.__args.device)
		print("Model loaded ...")


	def gen_cap(self, paths):
		input_images = []
		for path in paths:
			input_images.append(preprocess_image(path, self.__img_size))

		print("Generating captions ...\n")
		for i in range(len(input_images)):
			path = paths[i]
			image = input_images[i]
			image = image.to(device=self.__args.device)
			beam_search_kwargs = {'beam_size': self.__args.beam_size,
								'beam_max_seq_len': self.__args.max_seq_len,
								'sample_or_max': 'max',
								'how_many_outputs': 1,
								'sos_idx': self.__sos_idx,
								'eos_idx': self.__eos_idx}
			with torch.no_grad():
				pred, _ = self.__model(enc_x=image,
								enc_x_num_pads=[0],
								mode='beam_search', **beam_search_kwargs)
			pred = tokens2description(pred[0][0], self.__coco_tokens['idx2word_list'], self.__sos_idx, self.__eos_idx)
			print(path + ' \n\tDescription: ' + pred + '\n')
			return pred

	def client_handler(self, client_socket):
		with client_socket as sock:
			while True:
				request = sock.recv(1024)
				if not request:
					break
				print(f"Received: {request.decode('utf-8')}")
				cap = self.gen_cap([request.decode('utf-8')])
				sock.sendall(cap.encode('utf-8'))  # Echo back the received message

	def server(self):
		server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server_socket.bind(('0.0.0.0', 9999))
		server_socket.listen(5)  # 最大允许的连接数

		print("Server listening on port 9999")
		print("ctrl + c to stop server")

		try:
			while True:
				server_socket.settimeout(1.0)
				try:
					client_socket, addr = server_socket.accept()
					print(f"Accepted connection from {addr}")

					# 创建一个新的线程来处理客户端连接
					client_handler = threading.Thread(target=self.client_handler, args=(client_socket,))
					client_handler.start()
				except socket.timeout:
					# print('time out')
					continue

		except KeyboardInterrupt:
			print("KeyboardInterrupt received, closing server...")

		except:
			server_socket.close()
			print('server closed')



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Demo')
	parser.add_argument('--model_dim', type=int, default=512)
	parser.add_argument('--N_enc', type=int, default=3)
	parser.add_argument('--N_dec', type=int, default=3)
	parser.add_argument('--max_seq_len', type=int, default=74)
	parser.add_argument('--device', type=str, default='cpu')
	parser.add_argument('--load_path', type=str, default='./rf_model.pth')
	parser.add_argument('--image_paths', type=str,
						default=['./demo_material/tatin.jpg',
								 './demo_material/micheal.jpg',
								 './demo_material/napoleon.jpg',
								 './demo_material/cat_girl.jpg'],
						nargs='+')
	parser.add_argument('--beam_size', type=int, default=5)

	model = model_server(parser)
	model.server()

	print("Closed.")
