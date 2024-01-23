import logging
import time
# import datetime
import subprocess
from natsort import natsorted

import os
import json
import yaml




# def initialize_data_tree():

#     usr_shared_dir = os.path.join("usr", "shared")
#     if not os.path.exists(usr_shared_dir):
#         os.makedirs(usr_shared_dir)

#     objects_path = os.path.join(usr_shared_dir, "objects.json")
#     if not os.path.exists(objects_path):
#         json_io.save_json(objects_path, {})

#     scheduler_status_path = os.path.join(usr_shared_dir, "scheduler_status.json")
#     if not os.path.exists(scheduler_status_path):
#         json_io.save_json(scheduler_status_path, {})

#     public_image_sets_path = os.path.join("usr", "shared", "public_image_sets.json")
#     if not os.path.exists(public_image_sets_path):
#         json_io.save_json(public_image_sets_path, {})


#     usr_data_dir = os.path.join("usr", "data")
#     if not os.path.exists(usr_data_dir):
#         os.makedirs(usr_data_dir)





def app_init():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Waiting for postgres")

    postgres_up = False
    while not postgres_up:

        res = subprocess.run(["PGPASSWORD=secretDBkeypass", 
                        "psql", 
                        "-c", "'\l'", 
                        "-h", "db", 
                        "-p", "5432",
                        "-U", "agricounter_db_user",
                        "agricounter_db"
        ])

        if res.returncode != 0:
            time.sleep(1)
        else:
            postgres_up = True

    logger.info("Running migrations")
    subprocess.run(["npx", "sequelize-cli", "db-migrate"])

    logger.info("Running seeders")
    subprocess.run(["npx", "sequelize-cli", "db:seed:all"])

    logger.info("Starting npm")
    subprocess.run(["npm", "run", "start"])

        


#     echo 'Waiting for postgres'

# until PGPASSWORD=secretDBkeypass psql -c '\l' -h db -p 5432 -U agricounter_db_user agricounter_db; do
#   echo >&2 "$(date +%Y%m%dt%H%M%S) Postgres is unavailable - sleeping"
#   sleep 1
# done
# echo >&2 "$(date +%Y%m%dt%H%M%S) Postgres is up - executing command"



# echo 'running db:migrate'
# npx sequelize-cli db:migrate

# echo 'running db:seed:all'
# npx sequelize-cli db:seed:all


# echo 'Starting npm'
# npm run start








def configure():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args_path = os.path.join(".", "args.json")
    with open(args_path, 'r') as fp:
        args = json.load(fp)


    # logger.info("Cloning into plant_detection.git")
    # subprocess.run(["git", "clone", "https://github.com/eandvaag/plant_detection.git"])

    # logger.info("Cloning into plant_detection_viewer.git")
    # subprocess.run(["git", "clone", "https://github.com/eandvaag/plant_detection_viewer.git"])


    logger.info("Configuring docker-compose.yml")

    docker_compose_init_path = os.path.join("site", "docker-compose-init.yml")
    with open(docker_compose_init_path, 'r') as ymlfile:
        conf = yaml.safe_load(ymlfile)


    psql_env = conf["services"]["db"]["environment"]
    psql_env["POSTGRES_DB"] = args["postgres_db_name"]
    psql_env["POSTGRES_USER"] = args["postgres_db_username"]
    psql_env["POSTGRES_PASSWORD"] = args["postgres_db_password"]

    conf["services"]["db"]["ports"] = [str(args["postgres_db_port"]) + ":5432"]


    site_env = conf["services"]["myapp"]["environment"]
    site_env["DB_SCHEMA"] = args["postgres_db_name"]
    site_env["DB_USER"] = args["postgres_db_username"]
    site_env["DB_PASSWORD"] = args["postgres_db_password"]

    site_env["AC_PORT"] = args["site_port"]
    site_env["AC_PY_PORT"] = args["backend_python_port"]
    site_env["AC_PATH"] = args["url_prefix"]


    conf["services"]["myapp"]["ports"] = [str(args["site_port"]) + ":" + str(args["site_port"])]


    cwd = os.getcwd()
    site_volume = conf["services"]["myapp"]["volumes"][0]
    site_volume["source"] = os.path.join(cwd, "backend", "src", "usr")
    # site_volume["target"] = os.path.join("opt", "app", "site", "myapp", "usr")



    with open("docker-compose.yml", "w") as ymlfile:
        yaml.dump(conf, ymlfile, default_flow_style=False)


    logger.info("Writing seeders file")

    seeders_dir = os.path.join("site", "myapp", "seeders")
    # os.makedirs(seeders_dir, exist_ok=True)

    # d = datetime.datetime.now()

    seeders_name = "seed-users.js" #str(d.year) + str(d.month) + str(d.day) + str(d.hour) + str(d.minute) + "-seed-users.js"
    seeders_path = os.path.join(seeders_dir, seeders_name)

    f = open(seeders_path, "w")
    f.write(
        "'use strict';\n" +
        "\n" +
        "var bcrypt = require('bcrypt');\n" +
        "\n" +
        "module.exports = {\n" + 
        "    up: (queryInterface, Sequelize) => {\n" +
        "\n" +
        "        const salt = bcrypt.genSaltSync();\n" +
        "        return queryInterface.bulkInsert('users', [\n" +
        "            {\n" +
        "                username: '" + args["admin_username"] + "',\n" +
        "                password: bcrypt.hashSync('" + args["admin_password"] + "', salt),\n" +
        "                is_admin: true,\n"
        "                createdAt: new Date(),\n" +
        "                updatedAt: new Date()\n" +
        "            }"
    )

    for user in args["initial_users"]:
        f.write(
            ",\n" +
            "            {\n" +
            "                username: '" + user["username"] + "',\n" +
            "                password: bcrypt.hashSync('" + user["password"] + "', salt),\n" +
            "                is_admin: false,\n"
            "                createdAt: new Date(),\n" +
            "                updatedAt: new Date()\n" +
            "            }"
        )

    f.write(
        "\n" +
        "        ], {\n" +
        "        });\n" +
        "    },\n" +
        "    down: (queryInterface, Sequelize) => {\n" +
        "        return queryInterface.bulkDelete('users', null, {});\n" +
        "    }\n" +
        "};"
    )



    f.close()


    logger.info("Writing config.js")


    config_path = os.path.join("site", "myapp", "config", "config.js")

    f = open(config_path, "w")
    f.write(
        'module.exports = {\n' +
        '    "docker": {\n' +
        '        "username": "' + args["postgres_db_username"] + '",\n' +
        '        "password": "' + args["postgres_db_password"] + '",\n' +
        '        "database": "' + args["postgres_db_name"] + '",\n' +
        '        "host": "db",\n' +
        '        "dialect": "postgres",\n' +
        '        "port": 5432\n' +
        '    }\n' +
        '}'
    )
    f.close()



    logger.info("Writing objects.json")

    objects_path = os.path.join("backend", "src", "usr", "shared", "objects.json")
    initial_objects = natsorted(args["initial_objects"])
    objects = {"object_names": initial_objects}
    with open(objects_path, 'w') as fp:
        json.dump(objects, fp)



    logger.info("Starting Docker container")

    # subprocess.run(["docker-compose", "build", "-d"]) #"up", "-d", "--build"])
    subprocess.run(["docker-compose", "up", "-d"])

    # currently site/myapp/myapp-init.sh contains hardcoded names for the database and database user --> need to update
    # Dockerfile includes timezone -- this should be an argument in args.json



if __name__ == "__main__":

    configure()
