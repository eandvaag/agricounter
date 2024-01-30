To start app, move to plant_detection_viewer directory and run
```
    ./docker-ctl.sh start-headless
```

To stop app, move to plant_detection_viewer directory and run
```
    ./docker-ctl.sh stop
```


Docker Setup

install docker
install docker-compose

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list




create docker-compose-<CPU_NAME>.yml
run ./docker-ctl.sh start




Setup


Install node.
```
curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash - &&\
sudo apt-get install -y nodejs

sudo npm install -g n

sudo n 14.18.1
```

Install packages (execute from myapp directory).
```
npm install package.json
```

Install posgresql.
```
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

sudo apt-get update

sudo apt-get -y install postgresql
```

Acquire config.json and place in myapp/config.

Create database and database user.
```
sudo -u postgres psql
```
In psql (replace 'XXX' with password found in config.json):
```
CREATE DATABASE plant_detection_db;
CREATE ROLE plant_detection_dbuser WITH PASSWORD 'XXX';
ALTER ROLE "plant_detection_dbuser" WITH LOGIN;
```

Grant permissions to plant_detection_dbuser to create things inside the public schema
```
postgres=# \c plant_detection_db 
plant_detection_db=# GRANT ALL ON SCHEMA public TO plant_detection_dbuser;
```


Run migrations.
First cd to the myapp directory.
```
cd myapp
```

```
npx sequelize-cli db:migrate
```

Acquire user seeders file and place in myapp/seeders. Then run seeders.
```
npx sequelize-cli db:seed:all
```


Add environment variables to ~/.bashrc
```
export CC_IP="YOUR_IP_ADDRESS_HERE"
export CC_PORT="8110"
export CC_PY_PORT="8111"
export CC_PATH="/canola_counter_https"
```


Acquire cert.pem and key.pem and add to myapp directory.

Install image-magick.
```
sudo apt install imagemagick
```

Edit /etc/ImageMagick-6/policy.xml to allow larger files to be converted.
<policy domain="resource" name="disk" value="10GiB"/>


Install other apt packages.
```
sudo apt install libimage-exiftool-perl
sudo apt install libvips-tools
sudo apt install libgdal-dev gdal-bin
```

