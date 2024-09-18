import socket

def client():
	server_address = ('127.0.0.1', 9999)  # 服务器地址和端口
	message = ['./ExpansionNet_v2/demo_material/tatin.jpg',
				'./ExpansionNet_v2/demo_material/micheal.jpg',
				'./ExpansionNet_v2/demo_material/napoleon.jpg',
				'./ExpansionNet_v2/demo_material/cat_girl.jpg']

	# 创建一个 TCP/IP socket
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		# 连接到服务器
		sock.connect(server_address)
		

		for msg in message:
			# 发送数据
			print(f"Sending: {msg}")
			sock.sendall(msg.encode('utf-8'))
			
			# 接收回显数据
			response = sock.recv(1024)
			print(f"Received: {response.decode('utf-8')}")
		
		sock.close()

if __name__ == "__main__":
	client()
