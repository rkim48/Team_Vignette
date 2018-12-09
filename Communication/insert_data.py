import os
import time
import datetime
import glob
import MySQLdb

from time import strftime

# Variables for MySQL
db = MySQLdb.connect(host="localhost", user="root", password="root", db="vignette_db")
cur = db.cursor()

# Variables for TCP server
bind_ip = '10.214.179.146'
bind_port = 5005
BUFFER_SIZE = 1024

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((bind_ip, bind_port))
server.listen(1)  # max backlog of connections

conn, addr = server.accept()
print('Connection address:', addr)

# Store the data then close the connection
data = conn.recv(BUFFER_SIZE)
conn.close()

data = str(data)

data_list = data.split("\\n")
data_list = data_list[:len(data_list) - 1]
data_list[0] = data_list[0][2:]  # remove '0b' prefix on first entry

for i in range(len(data_list)):
    data_list[i] = data_list[i].split(",")

print("split data list:", data_list)
print()

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

# def readVals(filename):
#     file = open(filename, 'r')
#     lines = file.readlines()
#     file.close()

#     print(lines, '\n')
#     print(len(lines));
#     i = 0

#     for i in range(len(lines)):
#         lines[i] = lines[i].split(',')
#         # remove newline
#         lines[i][1] = lines[i][1][:len(lines[i][1]) - 1]

#     for line in lines:
#         print(line)

#     return lines;

entries = data_list
datetime = (time.strftime("%Y-%m-%d") + time.strftime(" | %H:%M:%S"));

print()
print(datetime)

# if the send is true, clear and write to the database
# else, print a message and close the DB without any changes
if send == True:
    clr = "DELETE FROM user_data"
    try:
        print("Clearing database...")
        cur.execute(clr)
        db.commit()
        print("Cleared\n")
    except:
        db.rollback()
        print("Failed clearing database")


    for entry in entries:
        # add data
        sql = ("INSERT INTO user_data(Time, Item, Frequency) VALUES(%s, %s, %s)", (datetime, entry[0], entry[1]))
        try:
            print("Writing to database...")
            # execute the sql command above (passing in a pointer)
            cur.execute(*sql)
            # Commit changes in the DB
            db.commit()
            print("Write complete!")
        except:
            # Roll back changes if there is an error
            db.rollback()
            print("Failed writing to database!")
else:
    print("Data not sent.")

cur.close()
db.close()
