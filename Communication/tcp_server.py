'''
Simple module for communication with a Raspberry PI on the same wireless
network using TCP sockets.
'''
import socket
import threading
import sys
import os

# Print message to inform user
print("Starting server...")
# IP Address of my PC
#bind_ip = '192.168.0.145'
bind_ip = '10.214.179.146'
bind_port = 5005
BUFFER_SIZE = 1024

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((bind_ip, bind_port))
server.listen(1)  # Max backlog of connections

conn, addr = server.accept()
print('Connection address:', addr)

data = conn.recv(BUFFER_SIZE)

# Close the server-side connection
conn.close()

# cast the data from bytes to string
data = str(data)

# split the data
data_list = data.split("\\n")
data_list = data_list[:len(data_list) - 1]
data_list[0] = data_list[0][2:]
# print("data list:", data_list)
# print()
for i in range(len(data_list)):
    data_list[i] = data_list[i].split(",")
    
# print()
print("split data list:", data_list)
print()

# ask the user if they want to send their data using 'input' until valid
# input is entered
while True:
    prompt = input("Do you want to send your data to P&G? [Y/n]: ")

    if prompt.lower() == 'n':
        send = False
        break
    elif prompt.lower() == 'y':
        send = True
        break
    else:
        print("Please type in Y or n!\n")

print("send:", send)
print()
# if send is true, then setup html file
# of course, we will change all this (including reading in the csv to
# utilize SQL to make it nice, but this is okay for now I guess)
html_file = open("index.html", "w")

# data is rendered through PHP and an sql database instead now so the
# code below doesn't really matter
# it does generate an index.html file that you can access in your browser
if send == True:
    # perform html file setup
    html_file.write("<!DOCTYPE html>")
    html_file.write("<html>")

    # writing the CSS
    html_file.write("<style>")
    html_file.write("table, th, td {")
    html_file.write("border: 1px solid black;")
    html_file.write("border-collapse: collapse;")
    html_file.write("}")
    html_file.write("</style>")

    # write each line as a new table entry in the file
    html_file.write("<table>")
    html_file.write("<caption>User Data</caption>")

    # use loop to write rows and columns into our table from the data
    # that we parsed (data_list)
    html_file.write("<tr>")
    html_file.write("<th>" + data_list[0][0] + "</th>")
    html_file.write("<th>" + data_list[0][1] + "</th>")
    html_file.write("</tr>")
    for i in range(1, len(data_list)):
        # for every element in our data list, add it to our html table as a 
        # row <tr> with two data columns that correspond to the entry
        # <td>
        html_file.write("<tr>")
        html_file.write("<td>" + data_list[i][0] + "</td>")
        html_file.write("<td>" + data_list[i][1] + "</td>")
        html_file.write("</tr>")

    html_file.write("</table>")
    html_file.write("</html>")

elif send == False:
    html_file.write("<!DOCTYPE html>")
    html_file.write("<html>")
    html_file.write("<body>")
    html_file.write("<h3>User has elected not to send their data.</h3>")
    html_file.write("</body>")
    html_file.write("</html>")

html_file.close()

#os.popen('http-server', 'w', 1)