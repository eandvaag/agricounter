
FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04
RUN apt update -y
RUN apt update && apt install -y exiftool
RUN apt update && apt install -y curl

ARG DEBIAN_FRONTEND=noninteractive
ARG AC_TIMEZONE
ENV TZ=${AC_TIMEZONE}
# ENV TZ=America/Regina
RUN apt-get install -y tzdata


RUN curl -fsSL https://deb.nodesource.com/setup_14.x | bash -
RUN apt install -y libgl1
RUN apt-get install -y libglib2.0-0
RUN apt update && apt-get install -y linux-libc-dev
RUN apt update && apt install -y nodejs
RUN apt update && apt install -y python3
RUN apt update && apt install -y python3-pip
RUN apt update && apt install -y iproute2
RUN apt update && apt install -y iputils-ping
RUN apt update && apt install -y dnsutils
RUN mkdir -p /opt/app
WORKDIR /opt/app
RUN adduser app
RUN usermod -a -G app app 

COPY --chown=app backend/src/requirements.txt ./backend/src/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --debug libclang
RUN pip3 install -r ./backend/src/requirements.txt
COPY --chown=app backend ./backend
COPY --chown=app site ./site


WORKDIR ./site/myapp
RUN npm install
RUN apt update && apt install -y postgresql-client
RUN apt update && apt install -y imagemagick
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN sed -i 's/policy domain="resource" name="disk" value="1GiB"/policy domain="resource" name="disk" value="8GiB"/' /etc/ImageMagick-6/policy.xml

RUN apt update && apt install -y libgdal-dev gdal-bin
RUN export gdal_ver=`gdalinfo --version | perl -a -e 'if ($_ =~ /.*\s+(\d+\.\d+\.\d+)/) { print "$1"; }'` && export max_ver=`pip index versions pygdal 2>/dev/null | grep -i 'Available versions' | perl -a -e '@fields=split(",",$_); $max = -1; foreach $field (@fields) { if ($field =~ /($ENV{'gdal_ver'}(\S+))/) { if ($2 > $max) { $max = $2; $max_ver = $1; } } } print "$max_ver\n";'` && pip3 install pygdal==$max_ver
RUN apt update && apt install -y libvips-tools

ARG CACHEBUST=1

USER app
RUN ln -s /opt/app/backend/src/usr /opt/app/site/myapp/usr


RUN chmod +x myapp-init.sh 
CMD /bin/sh /opt/app/site/myapp/myapp-init.sh
EXPOSE ${AC_PORT}
