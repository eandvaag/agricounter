const path = require('path');
const fs = require('fs');
const glob = require('glob');

const socket_io = require('socket.io');
const io = socket_io({
    "path": process.env.AC_PATH + "/socket.io"
});

let workspace_id_to_key = {};
let home_id_to_key = {};


io.on('connection', function(socket) {
	console.log('A user connected');

    socket.on("join_workspace", (key) => {
        console.log("join_workspace from", key);

        let occupied = false;
        for (let socket_id of Object.keys(workspace_id_to_key)) {
            if (workspace_id_to_key[socket_id] === key) {
                occupied = true;
                break;
            }
        }
        if (occupied) {
            io.to(socket.id).emit("workspace_occupied", {});
        }
        else {
            workspace_id_to_key[socket.id] = key;

            console.log("updated workspace_id_to_key", workspace_id_to_key);

            let [username, farm_name, field_name, mission_date] = key.split("/");

            let scheduler_status_path = path.join("usr", "shared", "scheduler_status.json")
            let scheduler_status;
            try {
                scheduler_status = JSON.parse(fs.readFileSync(scheduler_status_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
            }

            emit_image_set_status(username, farm_name, field_name, mission_date);
            emit_scheduler_status(scheduler_status);
        }

    });

    socket.on("join_home", (username) => {
        console.log("join_home from", username);

        home_id_to_key[socket.id] = username;

        console.log("updated home_id_to_key", home_id_to_key);
    });


    socket.on("disconnect", (reason) => {

        if (socket.id in home_id_to_key) {
            console.log("user disconnected from home");

            delete home_id_to_key[socket.id];

            console.log("updated home_id_to_key", home_id_to_key);
        }
        else if (socket.id in workspace_id_to_key) {
            console.log("user disconnected from workspace");

            delete workspace_id_to_key[socket.id];

            console.log("updated workspace_id_to_key", workspace_id_to_key);
        }
    });
});

function emit_image_set_status(username, farm_name, field_name, mission_date) {
    let key = username + "/" + farm_name + "/" + field_name + "/" + mission_date;


    let sel_socket_id = null;
    for (let socket_id of Object.keys(workspace_id_to_key)) {
        if (workspace_id_to_key[socket_id] === key) {
            sel_socket_id = socket_id;
            break;
        }
    }
    if (sel_socket_id !== null) {

        let image_set_dir = path.join("usr", "data", username, "image_sets",
                                        farm_name, field_name, mission_date);
        let model_dir = path.join(image_set_dir, "model");

        let status;
        let status_path = path.join(model_dir, "status.json");
        try {
            status = JSON.parse(fs.readFileSync(status_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
        }

        let training_dir = path.join(model_dir, "training");
        let prediction_dir = path.join(model_dir, "prediction");

        let num_outstanding;
        glob(path.join(prediction_dir, "image_requests", "*"), function(error, image_prediction_paths) {
            if (error) {
                console.log(error);
            }
            num_outstanding = image_prediction_paths.length;
            glob(path.join(prediction_dir, "image_set_requests", "pending", "*"), function(error, image_set_prediction_paths) {
                if (error) {
                    console.log(error);
                }
                num_outstanding = num_outstanding + image_set_prediction_paths.length;

                if (num_outstanding > 0) {
                    status["outstanding_prediction_requests"] = "True";
                }
                else {
                    status["outstanding_prediction_requests"] = "False";
                }

                status["usr_training_blocked"] = "True";
                let block_file_path = path.join(training_dir, "usr_block.json");
                try {
                    fs.accessSync(block_file_path, fs.constants.F_OK);
                }
                catch (e) {
                    status["usr_training_blocked"] = "False";
                }

                status["sys_training_blocked"] = "True";
                block_file_path = path.join(training_dir, "sys_block.json");
                try {
                    fs.accessSync(block_file_path, fs.constants.F_OK);
                }
                catch (e) {
                    status["sys_training_blocked"] = "False";
                }

                status["switch_request"] = "True";
                let switch_path = path.join(model_dir, "switch_request.json");
                try {
                    fs.accessSync(switch_path, fs.constants.F_OK);
                }
                catch (e) {
                    status["switch_request"] = "False";
                }
                    
                io.to(sel_socket_id).emit("image_set_status_change", status);

            });
        });
    }
}

function emit_scheduler_status(status) {

    let username = status["username"];
    let farm_name = status["farm_name"];
    let field_name = status["field_name"];
    let mission_date = status["mission_date"];

    emit_image_set_status(username, farm_name, field_name, mission_date);
    for (let socket_id of Object.keys(workspace_id_to_key)) {
        io.to(socket_id).emit("scheduler_status_change", status);
    }
}

exports.post_model_notification = function(req, res, next) {
    let username = req.body.username;

    model_notification(username);

    let response = {};
    response.message = "received";
    return res.json(response);
}

function model_notification(username) {

    console.log("model update occurred, sending to sockets");
    console.log("username", username);
    console.log("home_id_to_key", home_id_to_key);

    let key = username;

    for (let socket_id of Object.keys(home_id_to_key)) {
        if (home_id_to_key[socket_id] === key) {
            console.log("sending to socket", key);
            io.to(socket_id).emit("model_change", {});
        }
    }
}

exports.post_results_notification = function(req, res, next) {
    let username = req.body.username;
    let farm_name = req.body.farm_name;
    let field_name = req.body.field_name;
    let mission_date = req.body.mission_date;
    
    results_notification(username, farm_name, field_name, mission_date);

    let response = {};
    response.message = "received";
    return res.json(response);
    
}

function results_notification(username, farm_name, field_name, mission_date) {

    console.log("results update occurred, sending to sockets");
    console.log(username, farm_name, field_name, mission_date);
    console.log("home_id_to_key", home_id_to_key);

    let key = username;

    for (let socket_id of Object.keys(home_id_to_key)) {
        if (home_id_to_key[socket_id] === key) {
            io.to(socket_id).emit("results_change", {farm_name, field_name, mission_date});
        }
    }
}



exports.post_upload_notification = function(req, res, next) {
    let username = req.body.username;
    let farm_name = req.body.farm_name;
    let field_name = req.body.field_name;
    let mission_date = req.body.mission_date;

    console.log("upload update occurred, sending to sockets");
    console.log(username, farm_name, field_name, mission_date);
    console.log("home_id_to_key", home_id_to_key);

    let key = username;

    for (let socket_id of Object.keys(home_id_to_key)) {
        if (home_id_to_key[socket_id] === key) {
            io.to(socket_id).emit("upload_change", {farm_name, field_name, mission_date});
        }
    }

    let response = {};
    response.message = "received";
    return res.json(response);
}


exports.post_status_notification = function(req, res, next) {

    emit_scheduler_status(req.body);

    let response = {};
    response.message = "received";
    return res.json(response);
}


function dead_scheduler() {

    for (let socket_id of Object.keys(workspace_id_to_key)) {
        io.to(socket_id).emit("dead_scheduler", {});
    }

}







module.exports.io = io;
module.exports.workspace_id_to_key = workspace_id_to_key;
module.exports.results_notification = results_notification;
module.exports.dead_scheduler = dead_scheduler;