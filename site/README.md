


# AgriCounter Node.js Application


## Install (Non-Docker)


Install Node.js.
```
curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash - &&\
sudo apt-get install -y nodejs

sudo npm install -g n

sudo n 14.18.1
```

Install packages (execute from `site/myapp/` directory).
```
npm install package.json
```

Install PostGreSQL.
```
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

sudo apt-get update

sudo apt-get -y install postgresql
```

Acquire config.json and place in `site/myapp/config`.

Create database and database user.
```
sudo -u postgres psql
```
In psql (replace 'XXX' with password found in `config.json`):
```
CREATE DATABASE agricounter_db;
CREATE ROLE agricounter_db_user WITH PASSWORD 'XXX';
ALTER ROLE "agricounter_db_user" WITH LOGIN;
```

Grant privileges to agricounter_db_user.
```
postgres=# \c agricounter_db 
agricounter_db=# GRANT ALL ON SCHEMA public TO agricounter_db_user;
```


Run database migrations from the `site/myapp` directory:
```
npx sequelize-cli db:migrate
```

Acquire user seeders file and place in `site/myapp/seeders`. Then run seeders.
```
npx sequelize-cli db:seed:all
```


Add environment variables to `~/.bashrc`.
```
export AC_IP="YOUR_IP_ADDRESS_HERE"
export AC_PORT="8110"
export AC_PY_PORT="8111"
export AC_PATH="/agricounter"
export AC_API_KEY="YOUR_SECRET_API_KEY_HERE"
```


Acquire cert.pem and key.pem and add to `site/myapp` directory.


Install ImageMagick.
```
sudo apt install imagemagick
```

Edit /etc/ImageMagick-6/policy.xml to allow larger files to be converted.
```
    <policy domain="resource" name="disk" value="10GiB"/>
```



Install other apt packages.
```
sudo apt install libimage-exiftool-perl
sudo apt install libvips-tools
sudo apt install libgdal-dev gdal-bin
```

To start the Node.js application, execute the following command from the `site/myapp` directory:
```
	DEBUG=myapp:* npm start
```