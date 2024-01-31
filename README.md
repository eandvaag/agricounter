# AgriCounter
A web tool for performing object detection in agricultural images.



## Project Design

The AgriCounter tool is divided into two main components: a Node.js application and a Python process. These two processes run in parallel and communicate using a shared file tree and by sending and receiving http requests.

The `site` directory in this repository contains the Node.js application.

The `backend` directory contains the code for the Python process. The Python process is responsible for performing model-related tasks (training, fine-tuning, and prediction).


## Install

### Docker Install

`agricounter_ctl.py` contains four functions for creating and managing the AgriCounter tool inside a Docker container. After cloning the AgriCounter repository, a new Docker instance of AgriCounter can be built with the following Python commands:

```
    import agricounter_ctl
    agricounter_ctl.create()
```


`agricounter_ctl.create()` uses the arguments found in `args.json` to set up the application. Edit the `args.json` file in order to configure the setup as desired.


To stop the container without removing the PostGreSQL volume, use `agricounter_ctl.down()`. The container can then be re-built with `agricounter_ctl.up()`. To stop the container and remove the PostGreSQL volume, use `agricounter_ctl.destroy()`.




### Non-Docker Install

To set up the AgriCounter tool without Docker, the two main components of the project (located in the `site` and `backend` directories) need to be set up independently. Each of these directories contains a `README.md` file with instructions on how to perform the setup. Once the setup has been performed, the two processes should be started in two separate terminal sessions.


