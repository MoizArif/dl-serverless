printf "Enter CouchDB\'s Public IP: "
read publicIP
printf "Enter CouchDB\'s Username: "
read admin
printf "Enter CouchDB\'s Password: "
read password
echo $admin : $password @ $publicIP
