# AgriCounter
A web tool for performing object detection in agricultural images.



## Project Design

The AgriCounter tool is divided into two main components: a Node.js application and a Python process. These two processes run in parallel and communicate using a shared file tree and by sending and receiving http requests.

The `site` directory in this repository contains the Node.js application.

The `backend` directory contains the code for the Python process. The Python process is responsible for performing model-related tasks (training, fine-tuning, and prediction).


## Install

### Docker Install

`agricounter_ctl.py` can be used to create and manage the AgriCounter tool inside a Docker container. 
After cloning the AgriCounter repository, a new Docker instance of AgriCounter can be built with the following command:

```
./agricounter_ctl.py --create
```

In order to set up the application, `agricounter_ctl.py` will attempt to read a configuration file called `args.json`. The `args.json` file should be located in the root directory of the AgriCounter repository.

Included in this repository is an `args-template.json` file with the required configuration keys. Before running `./agricounter_ctl.py --create`, this file needs to be edited with the desired configuration values and renamed to `args.json`. 

Below is an explanation of the keys that `agricounter_ctl.py` expects to find in the `args.json` file:

- `url_prefix`: All URL paths will begin with this string. Example: "/agricounter"
- `site_port`: The port number that the Node.js application will listen on.
- `backend_python_port`: The port number that the Python server process will listen on.
- `postgres_db_name`: The name of the PostGres database that AgriCounter will use to store user accounts.
- `postgres_db_username`: The username for the PostGres role that has database privileges.
- `postgres_db_password`: The password for the PostGres role that has database privileges.
- `postgres_db_port`: The port number that the PostGres database will listen on.
- `api_key`: The Python process sends this password whenever it sends an HTTP request to the Node.js application. The Node.js application uses this password to verify that the request was sent by the Python process.
- `admin_username`: Username for the site's administrator account.
- `admin_password`: Password for the site's administrator account.
- `gpu_index`: Index of GPU device to use. If only one GPU is available, this should be 0. Use -1 if you want to use the CPU instead.
- `use_slurm`: Use SLURM job scheduler (for HPC environments). Requires setup of a `slurm_config.json` file in `backend/src`.

To stop the Docker container without removing the PostGreSQL volume, use `./agricounter_ctl.py --down`. The container can then be rebuilt with `./agricounter_ctl.py --up`. To stop the container and remove the PostGreSQL volume, use `./agricounter_ctl.py --destroy`.

When a AgriCounter Docker instance is created with `./agricounter_ctl.py -c`, only the administrator account 
is seeded to the database. In order to add regular user accounts, it is necessary to log in as the 
administrator and add the new user accounts through the administrator web page interface. It is also necessary to 
specify the object classes that the AgriCounter tool will be used to detect (e.g., "Canola Seedling"). This is also 
done from the administrator's web page.


### Non-Docker Install

To set up the AgriCounter tool without Docker, the two main components of the project (located in the `site` and `backend` directories) need to be set up independently. Each of these directories contains a `README.md` file with instructions on how to perform the setup. Once the setup has been performed, the two processes should be started in two separate terminal sessions.


