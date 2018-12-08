'''
Sending data to host through TCP. Will upgrade to web interface soon.
'''
import socket

print("Starting client...")

# bind ip is the ip address of the server (my laptop in this case)
bind_ip = '10.214.179.146'
bind_port = 5005
BUFFER_SIZE = 1024

# open connection
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((bind_ip, bind_port))

# open csv file and send its contents to the server
filename = 'dummy.csv'
f = open(filename, 'r')
line = f.read(BUFFER_SIZE)
print(line)
client.send(line.encode())
    
print("messages sent!")
client.close()
