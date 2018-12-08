Vignette Communication Module Info:
- 'tcp_client.py' is on the PI and 'tcp_server.py' should be on your laptop/an actual web server
- Outgoing data is sent from 'tcp_client.py' to the same port (5005) opened
  on a laptop (set up on my laptop)
- bind_ip is the IPv4 address of your webserver/laptop
    - to find this:
    1) type 'ifconfig' into terminal (Mac/Linux) on your laptop
    2) I think it's 'ipconfig' on Windows
    3) Pick the non LOCALHOST ip (so NOT 127.0.0.1 or 0.0.0.1 or 10.0.0.1), 
    but rather the ip broadcasted by your Wireless Network Adapter

Walkthrough code in 'tcp_client.py':
    1) client.connect((bind_ip, bind_port))
    - opens a TCP connection to the given IP address on the given port

    # open csv file and sent its contents to the server
    2) filename = 'dummy.csv'
    - filename of file you want to open i.e. 'classification_data.csv'
    - format of .csv file should be comma delimited key-value pairs separated
      by newline characters (check this by opening the .csv in a plain text 
      editor or by typing 'cat classification_data.csv' into terminal), i.e.:

      key1,value1\n
      key2,value2\n
      key3,value3\n
      key4,value4\n
      ...

    3) f = open(filename, 'r')
    - open the csv file

    4) line = f.read(BUFFER_SIZE)
    - read the csv file in (set BUFFER_SIZE according to your needs, this can
      be changed later)

    #print(line)
    5) client.send(line.encode())
    - encode the csv data to UTF-8 and send it to the TCP server

    6) print("messages sent!")
       client.close()
    - print confirmation message and close connection

Steps to get set up:
1) Put the code in 'tcp_client.py' in the same folder as where the .csv will be.
2) Put 'import tcp_client' at the very top
3) Put 'import socket' at the very top 
   NOTE: may need [pip3 install socket / pip install socket] if your pi doesn't 
      find the socket module 
4) If the messages weren't sent, the code will fail at the line:
   - client.connect((bind_ip, bind_port)) 
   - error message will be along the lines of: could not open connection or it may
   - not show an error at all
   - this error will be the case when there is no TCP server on the receiving
     end

Optional, but recommended:
1) Setup 'tcp_server.py' code on your local laptop to test it out
   NOTE: 'tcp_server.py' is the non-SQL version of what I demo'd (first time).
          This is so that you will not be required to install mySQL, PHP, and
          Apache. However, the socket-to-socket communication will be done 
          exactly the same way. 

2) Put in the IP of your laptop computer as explained in the beginning.

3) If it works you'll get some friendly messages on your laptop's terminal/
python IDE console printing out the transferred key values

4) You'll then get a message prompting if you want it to send the data to 
P&G. Type [Y/n].

4) These key-values are what I am pushing into my SQL DB and then will be
asking the user if they want to send their data in my SQL python code. 
The PHP will handle rendering of the appropriate webpage. 

5) This code does generate an 'index.html' file on your laptop in the same
directory though. If you want to see the different responses to 

'Do you want to send your data to P&G?', just do the following:
    Type the below into terminal:
    - 1) npm install http-server -g
    Once it is installed, type:
    - 2) http-server index.html

    Then, open your localhost url/whatever server IP it says that index.html
    is being hosted on.

6) Blow your fuckin load because this project will make us millions.

7) Move to Silicon Valley and turn this project into a company.

8) Get acquisition offer, but then turn it down in the hope of making billions.

9) Reenact the story in the show 'Silicon Valley'.

10) Make hundreds of millions, as a compromise.
