import socket

def check_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    if result == 0:
        print(f"Port {port} is open")
    else:
        print(f"Port {port} is not open")
    sock.close()

check_port(8080)
