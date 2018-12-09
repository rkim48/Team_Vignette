How to install LAMP Stack (Linux, Apache, MySQL, and PHP):
1) Make sure you're on Linux (Ubuntu preferred)
2) Install Apache
    - sudo apt install apache2
3) Install MySQL
    - sudo apt install mysql-server
4) Install PHP
    - sudo apt install php-pear php-fpm php-dev php-zip php-curl php-xmlrpc php-gd php-mysql php-mbstring php-xml libapache2-mod-php

5) Restart Server
    - sudo systemctl restart apache2.service

Commands for Apache server (hosted on 'localhost' by default):
1) Start Apache server:
    - sudo systemctl start apache2.service
2) Check status of Apache server:
    - sudo systemctl status apache2.service
3) Restart Apache server:
    - sudo systemctl restart apache2.service

PHP files:
- PHP files are stored in /var/www/html

Install the necessary Python for MySQL:
1) Get pip and pip3 if you dont have it:
    - sudo apt-get install pip
    - sudo apt-get install pip3

2) Install mysql client on Python 3
    - pip3 install mysql client

Setting up MySQL Database:
1) 