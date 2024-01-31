import logging
import subprocess
from natsort import natsorted

import os
import json
import yaml




def create():

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args_path = os.path.join(".", "args.json")
    with open(args_path, 'r') as fp:
        args = json.load(fp)


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



    with open("docker-compose.yml", "w") as ymlfile:
        yaml.dump(conf, ymlfile, default_flow_style=False)


    logger.info("Writing seeders file")

    seeders_dir = os.path.join("site", "myapp", "seeders")

    seeders_name = "seed-users.js"
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

    subprocess.run(["docker-compose", "up", "-d"])

    # TODO: Dockerfile includes timezone -- this should be an argument in args.json




def resume():
    subprocess.run(["docker-compose", "up", "-d"])

def stop():
    subprocess.run(["docker-compose", "down", "--rmi", "local"])

def destroy():
    subprocess.run(["docker-compose", "down", "-v", "--rmi", "local"])

