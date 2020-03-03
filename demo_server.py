import scipy.io as sio # to save detection as mat files
from PIL import ImageDraw,Image
import socket
import sys
import cv2
import struct ## new
import zlib
import pickle,os
import numpy as np
import argparse
from data import UCF24_CLASSES
import matplotlib.pyplot as plt
CLASSES = UCF24_CLASSES 

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Demo Server')
parser.add_argument('--showImage', default=True, type=str2bool, help='Show event detection image')

args = parser.parse_args()

if __name__ == '__main__':
	HOST=''
	PORT=8485

	s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	# print('Socket created')

	s.bind((HOST,PORT))
	# print('Socket bind complete')
	s.listen(10)
	print('Server is up')

	conn,addr=s.accept()

	data = b""
	payload_size = struct.calcsize(">L")

	output_dir = '/home/bo/research/dataset/ucf24/detections'
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	dtind = 0

	while True:
		while len(data) < payload_size:
			data += conn.recv(4096)

		packed_msg_size = data[:payload_size]
		data = data[payload_size:]
		msg_size = struct.unpack(">L", packed_msg_size)[0]

		while len(data) < payload_size:
			data += conn.recv(4096)

		packed_class_label = data[:payload_size]
		data = data[payload_size:]
		class_label = struct.unpack(">L", packed_class_label)[0]

		while len(data) < payload_size:
			data += conn.recv(4096)

		packed_gt_label = data[:payload_size]
		data = data[payload_size:]
		gt_label = struct.unpack(">L", packed_gt_label)[0]
        
		while len(data) < msg_size:
			data += conn.recv(4096)
		frame_data = data[:msg_size]
		data = data[msg_size:]

		frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
		frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

		output_tube_dir = output_dir + '/event-' + \
        				str(dtind) + '-' + \
        				CLASSES[class_label] + '(' + CLASSES[gt_label] + ').png'
		dtind += 1

		img = Image.fromarray(frame.astype(np.uint8))
		img.save(output_tube_dir)

		print('Event detected:',CLASSES[class_label],'(',CLASSES[gt_label],')',
        				'Success' if class_label==gt_label else 'Failure')
		
		if args.showImage:
			b,g,r = cv2.split(frame)
			frame_rgb = cv2.merge((r,g,b))

			cv2.imshow('Event detection result',frame_rgb)
			cv2.waitKey(1)