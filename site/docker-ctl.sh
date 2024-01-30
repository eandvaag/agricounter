#/bin/bash

if [[ "$#" -ne 2 ]]; then
	echo "Usage: docker-ctl [CPU_NAME] [CMD]"
	exit 2
fi

if [[ "$2" = "start" ]] || [[ "$2" = "start-headless" ]]; then
	cp docker-compose-"$1".yml ../docker-compose.yml
	cp Dockerfile ..
	cp .dockerignore ..
	cd ..
	if [ "$2" = "start" ]; then
		export CACHEBUST=$(date +%s) && docker-compose up
	else
		docker-compose up -d
	fi

elif [[ "$2" = "stop" ]]; then
	cd ..
	docker-compose down -v --rmi local

else
	echo "Invalid command argument"

fi

