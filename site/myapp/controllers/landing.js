const path = require('path');
const { v4: uuidv4 } = require('uuid');
const fs = require('fs');
const nat_orderBy = require('natural-orderby');
const { spawn, exec } = require('child_process');
const http = require('http');

const { Op } = require("sequelize");
const models = require('../models');

const glob = require("glob");


var socket_api = require('../socket_api');


const USR_DATA_ROOT = path.join("usr", "data");
const USR_SHARED_ROOT = path.join("usr", "shared");

let active_uploads = {};
var Mutex = require('async-mutex').Mutex;


const image_sets_mutex = new Mutex();
const camera_mutex = new Mutex();



const MAX_EXTENSIONLESS_FILENAME_LENGTH = 100;

const FILE_FORMAT =               /[\s `!@#$%^&*()+\=\[\]{};':"\\|,<>\/?~]/;
const FARM_FIELD_MISSION_FORMAT = /[\s `!@#$%^&*()+\=\[\]{}.;':"\\|,<>\/?~]/;
const MODEL_NAME_FORMAT =         /[\s `!@#$%^&*()+\=\[\]{}.;':"\\|,<>\/?~]/;
const OBJECT_NAME_FORMAT =        /[`!@#$%^&*()+\=\[\]{}.;':"\\|,<>\/?~]/;
const USERNAME_FORMAT =           /[\s `!@#$%^&*()+\=\[\]{}.;':"\\|,<>\/?~]/;


const MIN_CAMERA_HEIGHT = 0.01;
const MAX_CAMERA_HEIGHT = 1000000000;

const allowed_hotkeys = [
    "Tab", "Caps Lock", "Shift", "Control", "Alt", "Delete", 
    " ", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown",
    "-", "+", "Backspace", "[", "]", "Enter", ";", "'",
    "\\", ",", ".", "/", "`",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", 
    "q", "w", "e", "r", "t", "y", "u", "i", "o", "p", 
    "a", "s", "d", "f", "g", "h", "j", "k", "l", 
    "z", "x", "c", "v", "b", "n", "m"
];

Date.prototype.isValid = function () {
    // An invalid date object returns NaN for getTime() and NaN is the only
    // object not strictly equal to itself.
    return this.getTime() === this.getTime();
};  


if (process.env.NODE_ENV === "docker") {
    console.log("Starting the python server...");

    let scheduler = spawn("python3", ["../../backend/src/server.py"]);


    scheduler.on('close', (code) => {
        console.log(`Scheduler closed with code ${code}.`);
    });

    scheduler.on('exit', (code) => {
        console.log(`Scheduler exited with code ${code}.`);
    });

    scheduler.stderr.on('data', (data) => {
        console.error(`Scheduler stderr: ${data}`);
    });
    scheduler.stdout.on('data', (data) => {
        console.log(`Scheduler stdout: ${data}`);
    });

    scheduler.on('SIGINT', function() {
        console.log('Scheduler received SIGINT signal');
    });

    scheduler.on('error', (error) => {
        console.log("Failed to start scheduler subprocess.");
        console.log(error);
    });

}


exports.get_sign_in = function(req, res, next) {
    res.render('sign_in');
}


exports.post_sign_in = function(req, res, next) {
    let response = {};
    response.not_found = false;
    response.error = false;
    response.maintenance = false;

    return models.users.findOne({
    where: {
        username: req.body.username,
    }
    }).then(user => {
        if (!user) {
            response.not_found = true;
            return res.json(response);
        }
        else {
            if (!user.check_password(req.body.password)) {
                response.not_found = true;
                return res.json(response);
            }
            else {
                if (user.is_admin) {
                    req.session.user = user.dataValues;
                    response.redirect = process.env.AC_PATH + "/admin";
                    return res.json(response);
                }
                else {
                    req.session.user = user.dataValues;
                    response.redirect = process.env.AC_PATH + "/home/" + req.body.username;
                    return res.json(response);
                }
            }
        }
    }).catch(error => {
        console.log(error);
        response.error = true;
        return res.json(response);
    });
}


function get_subdirnames(dir) {
    let subdirnames = [];
    let list = fs.readdirSync(dir);
    list.forEach(function(file) {
        let fpath = path.join(dir, file);
        let stat = fs.statSync(fpath);
        if (stat && stat.isDirectory()) {
            subdirnames.push(file);
        }
    });
    return subdirnames;
}

function get_subdirpaths(dir) {
    let subdirpaths = [];
    let list = fs.readdirSync(dir);
    list.forEach(function(file) {
        let fpath = path.join(dir, file);
        let stat = fs.statSync(fpath);
        if (stat && stat.isDirectory()) {
            subdirpaths.push(fpath);
        }
    });
    return subdirpaths;
}

function fpath_exists(fpath) {
    let exists = true;
    try {
        fs.accessSync(fpath, fs.constants.F_OK);
    }
    catch (e) {
        exists = false;
    }
    return exists;
}


exports.get_admin = function(req, res, next) {

    if (req.session.user && req.cookies.user_sid) {
                
        let objects_path = path.join(USR_SHARED_ROOT, "objects.json");
        let objects;
        try {
            objects = JSON.parse(fs.readFileSync(objects_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }

        return models.users.findAll({
            where: {
                username: {
                    [Op.not]: "admin"
                }
            }
        }).then(users => {

            let data = {};
            data["users"] = users;
            data["object_names"] = objects["object_names"];
            
            res.render("admin", {username: req.session.user.username, data: data});

        }).catch(error => {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        });
    }
    else {
        return res.redirect(process.env.AC_PATH);
    }
}



function init_usr(username) {

    let usr_dirs = [
        path.join(USR_DATA_ROOT, username),
        path.join(USR_DATA_ROOT, username, "cameras"),
        path.join(USR_DATA_ROOT, username, "image_sets"),
        path.join(USR_DATA_ROOT, username, "models"),
        path.join(USR_DATA_ROOT, username, "models", "pending"),
        path.join(USR_DATA_ROOT, username, "models", "aborted"),
        path.join(USR_DATA_ROOT, username, "models", "available"),
        path.join(USR_DATA_ROOT, username, "models", "available", "public"),
        path.join(USR_DATA_ROOT, username, "models", "available", "private")
    ];


    for (let usr_dir of usr_dirs) {
        try {
            fs.mkdirSync(usr_dir, { recursive: true });
        }
        catch(error) {
            return false;
        }
    }

    let init_cameras_path = path.join(USR_SHARED_ROOT, "init_cameras.json");
    let cameras_path = path.join(USR_DATA_ROOT, username, "cameras", "cameras.json");
    try {
        fs.copyFileSync(init_cameras_path, cameras_path, fs.constants.COPYFILE_EXCL);
    }
    catch (error) {
        return false;
    }

    let private_image_sets_path = path.join(USR_DATA_ROOT, username, "private_image_sets.json");
    try {
        fs.writeFileSync(private_image_sets_path, JSON.stringify({}));
    }
    catch (error) {
        return false;
    }

    let default_overlay_apperance_path = path.join(USR_SHARED_ROOT, "default_overlay_appearance.json");
    let overlay_appearance_path = path.join(USR_DATA_ROOT, username, "overlay_appearance.json");
    try {
        fs.copyFileSync(default_overlay_apperance_path, overlay_appearance_path, fs.constants.COPYFILE_EXCL);
    }
    catch (error) {
        return false;
    }

    let default_hotkeys_path = path.join(USR_SHARED_ROOT, "default_hotkeys.json");
    let hotkeys_path = path.join(USR_DATA_ROOT, username, "hotkeys.json");
    try {
        fs.copyFileSync(default_hotkeys_path, hotkeys_path, fs.constants.COPYFILE_EXCL);
    }
    catch (error) {
        return false;
    }

    return true;
}


exports.post_admin = function(req, res, next) {


    let action = req.body.action;
    let response = {};

    if (action == "add_object_class") {

        const MIN_OBJECT_NAME_LENGTH = 1;
        const MAX_OBJECT_NAME_LENGTH = 255;


        let object_name = req.body.object_name;

        if (OBJECT_NAME_FORMAT.test(object_name)) {
            response.message = "The provided object name contains illegal characters.";
            response.error = true;
            return res.json(response);
        }

        if (object_name.length < MIN_OBJECT_NAME_LENGTH) {
            response.message = "The provided object name is too short.";
            response.error = true;
            return res.json(response);
        }

        if (object_name.length > MAX_OBJECT_NAME_LENGTH) {
            response.message = "The provided object name is too long.";
            response.error = true;
            return res.json(response);
        }
                
        let objects_path = path.join(USR_SHARED_ROOT, "objects.json");
        let objects;
        try {
            objects = JSON.parse(fs.readFileSync(objects_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            response.message = "Failed to read objects file";
            response.error = true;
            return res.json(response);
        }

        let object_names = objects["object_names"];

        if (object_names.includes(object_name)) {
            response.message = "The provided object name has already been added to the system.";
            response.error = true;
            return res.json(response);
        }


        object_names.push(object_name);
        object_names = nat_orderBy.orderBy(object_names);


        objects["object_names"] = object_names;

        try {
            fs.writeFileSync(objects_path, JSON.stringify(objects));
        }
        catch (error) {
            console.log(error);
            response.message = "Failed to write object classes.";
            response.error = true;
            return res.json(response);
        }

        response.error = false;
        return res.json(response);
    }


    else if (action === "create_user_account") {


        const MIN_USERNAME_LENGTH = 1;
        const MAX_USERNAME_LENGTH = 255;


        const MIN_PASSWORD_LENGTH = 1;
        const MAX_PASSWORD_LENGTH = 255;

        
        let username = req.body.username;
        let password = req.body.password;


        if ((typeof username !== 'string') && (!(username instanceof String))) {
            response.message = "The provided username is not a string.";
            response.error = true;
            return res.json(response);
        }

        if ((typeof password !== 'string') && (!(password instanceof String))) {
            response.message = "The provided password is not a string.";
            response.error = true;
            return res.json(response);
        }

        if (USERNAME_FORMAT.test(username)) {
            response.message = "The provided username contains illegal characters.";
            response.error = true;
            return res.json(response);
        }

        if (username.length < MIN_USERNAME_LENGTH) {
            response.message = "The provided username is too short.";
            response.error = true;
            return res.json(response);
        }

        if (username.length > MAX_USERNAME_LENGTH) {
            response.message = "The provided username is too long.";
            response.error = true;
            return res.json(response);
        }

        if (password.length < MIN_PASSWORD_LENGTH) {
            response.message = "The provided password is too short.";
            response.error = true;
            return res.json(response);
        }

        if (password.length > MAX_PASSWORD_LENGTH) {
            response.message = "The provided password is too long.";
            response.error = true;
            return res.json(response);
        }


        return models.users.findOne({
            where: {
                username: username
            }
        }).then(user => {
            if (user) {
                response.message = "The provided username is in use by an existing account.";
                response.error = true;
                return res.json(response);
            }
            else {
                    
                let dirs_initialized = init_usr(username);
                
                if (dirs_initialized) {
                    return models.users.create({
                        username: req.body.username,
                        password: req.body.password,
                        is_admin: false
                    }).then(user => {

                        response.error = false;
                        return res.json(response);
                    }).catch(error => {
                        console.log(error);
                        let usr_dir = path.join(USR_DATA_ROOT, username);
                        try {
                            fs.rmSync(usr_dir, { recursive: true, force: false });
                        }
                        catch(error) {
                            console.log(error);
                        }
                        response.message = "An error occurred while creating the user account.";
                        response.error = true;
                        return res.json(response);
                    });

                }
                else {
                    response.message = "An error occurred while initializing the user's directory tree.";
                    response.error = true;
                    return res.json(response);
                }

            }
        }).catch(error => {
            console.log(error);
            response.message = "An error occurred while checking for user account uniqueness.";
            response.error = true;
            return res.json(response);
        });

    }
}

exports.get_home = function(req, res, next) {

    if ((req.session.user && req.cookies.user_sid) && (req.params.username === req.session.user.username)) {

        let username = req.session.user.username;

        let image_sets_data = {};
        let image_sets_root = path.join(USR_DATA_ROOT, username, "image_sets");
        let farm_names;
        try {
           farm_names = get_subdirnames(image_sets_root);
        }
        catch (error) {
            return res.redirect(process.env.AC_PATH);
        }

        let overlay_appearance;
        let overlay_appearance_path = path.join(USR_DATA_ROOT, username, "overlay_appearance.json");
        try {
            overlay_appearance = JSON.parse(fs.readFileSync(overlay_appearance_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }


        let maintenance_time = "";
        let maintenance_path = path.join(USR_SHARED_ROOT, "maintenance.json");
        if (fpath_exists(maintenance_path)) {
            try {
                maintenance_log = JSON.parse(fs.readFileSync(maintenance_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            maintenance_time = maintenance_log["maintenance_time"];
        }


        for (let farm_name of farm_names) {
            
            let farm_root = path.join(image_sets_root, farm_name);
            let field_names = get_subdirnames(farm_root);

            for (let field_name of field_names) {
                let field_root = path.join(farm_root, field_name);
                let mission_dates = get_subdirnames(field_root);
                
                for (let mission_date of mission_dates) {
                    let mission_root = path.join(field_root, mission_date);

                    let upload_status_path = path.join(mission_root, "upload_status.json");
                    if (fs.existsSync(upload_status_path)) {

                        try {
                            upload_status = JSON.parse(fs.readFileSync(upload_status_path, 'utf8'));
                        }
                        catch (error) {
                            return res.redirect(process.env.AC_PATH);
                        }

                        if (!(farm_name in image_sets_data)) {
                            image_sets_data[farm_name] = {};
                        }
                        if (!(field_name in image_sets_data[farm_name])) {
                            image_sets_data[farm_name][field_name] = {};
                        }

                        image_sets_data[farm_name][field_name][mission_date] = upload_status;
                    }

                    //else {
                        /* Upload was never completed, remove the directory */
                        //fs.rmSync(mission_root, { recursive: true, force: true });
/*
                        let missions = get_subdirs(field_dir);
                        if (missions.length == 0) {
                            fs.rmSync(field_dir, { recursive: true, force: true });
                            let fields = get_subdirs(farm_dir);
                            if (fields.length == 0) {
                                fs.rmSync(farm_dir, { recursive: true, force: true });
                            }
                        }*/
                    //}
                }
            }
        }


        image_sets_mutex.acquire()
        .then(function(release) {
            let public_image_sets_path = path.join(USR_SHARED_ROOT, "public_image_sets.json");
            let available_image_sets;
            try {
                available_image_sets = JSON.parse(fs.readFileSync(public_image_sets_path, 'utf8'));
            }
            catch (error) {
                release();
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            let private_image_sets_path = path.join(USR_DATA_ROOT, username, "private_image_sets.json");
            let private_image_sets;
            try {
                private_image_sets = JSON.parse(fs.readFileSync(private_image_sets_path, 'utf8'));
            }
            catch (error) {
                release();
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            release();

            for (let image_set_key of Object.keys(private_image_sets)) {
                available_image_sets[image_set_key] = private_image_sets[image_set_key];
            }

                
            let objects_path = path.join(USR_SHARED_ROOT, "objects.json");
            let objects;
            try {
                objects = JSON.parse(fs.readFileSync(objects_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }
            


            camera_mutex.acquire()
            .then(function(release) {

                let camera_specs;
                let camera_specs_path = path.join(USR_DATA_ROOT, username, "cameras", "cameras.json");
                try {
                    camera_specs = JSON.parse(fs.readFileSync(camera_specs_path, 'utf8'));
                }
                catch (error) {
                    release();
                    console.log(error);
                    return res.redirect(process.env.AC_PATH);
                }


                release();

                let data = {};
                data["username"] = username;
                data["image_sets_data"] = image_sets_data;
                data["camera_specs"] = camera_specs;
                data["objects"] = objects;
                data["available_image_sets"] = available_image_sets;
                data["overlay_appearance"] = overlay_appearance;
                data["maintenance_time"] = maintenance_time;
            
                res.render("home", {
                    username: username, 
                    data: data
                });


            }).catch(function(error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            });

        }).catch(function(error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        });
    }
    else {
        return res.redirect(process.env.AC_PATH);
    }
}

function is_hex_color(hex_color) {
    if (hex_color.length != 7) {
        return false;
    }
    if (hex_color[0] !== "#") {
        return false;
    }
    let valid_chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F"];
    for (let i = 1; i < hex_color.length; i++) {
        if (!(valid_chars.includes(hex_color[i]))) {
            return false;
        }
    }

    return true;
}

exports.post_hotkey_change = function(req, res, next) {
    let response = {};

    let hotkeys = JSON.parse(req.body.hotkeys);
    let hotkeys_path = path.join(USR_DATA_ROOT, req.session.user.username, "hotkeys.json");

    let hotkey_vals = Object.values(hotkeys);
    for (let hotkey_val of hotkey_vals) {
        if (!(allowed_hotkeys.includes(hotkey_val))) {
            response.error = true;
            response.message = "Unrecognized hotkey value: '" + hotkey_val + "'.";
            return res.json(response);
        }
    }
    let unique_hotkey_vals = [... new Set(hotkey_vals)];
    if (hotkey_vals.length > unique_hotkey_vals.length) {
        response.error = true;
        response.message = "Submitted hotkey values are not unique.";
        return res.json(response);
    }

    try {
        fs.writeFileSync(hotkeys_path, JSON.stringify(hotkeys));
    }
    catch (error) {
        response.error = true;
        response.message = "Failed to save hotkeys.";
        return res.json(response);
    }

    response.error = false;
    return res.json(response);
}


exports.post_overlay_appearance_change = function(req, res, next) {

    let overlay_appearance = JSON.parse(req.body.overlay_appearance);
    let overlay_appearance_path = path.join(USR_DATA_ROOT, req.session.user.username, "overlay_appearance.json");

    let response = {};
    for (let k of ["draw_order", "style", "colors"]) {
        if (!(k in overlay_appearance)) {
            response.error = true;
            response.message = "Invalid overlay colors object.";
            return res.json(response);
        }
    }


    if (((overlay_appearance["draw_order"].length != 5) || 
        (Object.keys(overlay_appearance["style"]).length != 5)) || 
        (Object.keys(overlay_appearance["colors"]).length != 5)) {
     
        response.error = true;
        response.message = "Invalid overlay colors object.";
        return res.json(response);
    }

    let overlay_keys = ["annotation", "prediction", "region_of_interest", "fine_tuning_region", "test_region"];
    for (let overlay_key of overlay_keys) {

        if (!(overlay_appearance["draw_order"]).includes(overlay_key)) {
            response.error = true;
            response.message = "Missing overlay key: '" + overlay_key + "'.";
            return res.json(response);
        }

        if (!(overlay_key in overlay_appearance["style"])) {
            response.error = true;
            response.message = "Missing overlay key: '" + overlay_key + "'.";
            return res.json(response);
        }
        let style = overlay_appearance["style"][overlay_key];
        if (style !== "strokeRect" && style !== "fillRect") {
            response.error = true;
            response.message = "Invalid style key in overlays object.";
            return res.json(response);
        }


        if (!(overlay_key in overlay_appearance["colors"])) {
            response.error = true;
            response.message = "Missing overlay key: '" + overlay_key + "'.";
            return res.json(response);
        }
        let colors_to_test;
        if ((overlay_key === "annotation") || (overlay_key === "prediction")) {
            if (!(Array.isArray(overlay_appearance["colors"][overlay_key]))) {
                response.error = true;
                response.message = "Invalid overlay colors object.";
                return res.json(response);
            }
            colors_to_test = overlay_appearance["colors"][overlay_key];
        }
        else {
            colors_to_test = [overlay_appearance["colors"][overlay_key]];
        }
        for (let color_to_test of colors_to_test) {
            if (!(is_hex_color(color_to_test))) {
                response.error = true;
                response.message = "Invalid overlay color provided.";
                return res.json(response);
            }
        }

    }

    try {
        fs.writeFileSync(overlay_appearance_path, JSON.stringify(overlay_appearance));
    }
    catch (error) {
        response.error = true;
        response.message = "Failed to save overlay colors.";
        return res.json(response);
    }

    response.error = false;
    return res.json(response);


}

exports.get_workspace = function(req, res, next) {

    if ((req.session.user && req.cookies.user_sid) && (req.params.username === req.session.user.username)) {
        
        let username = req.session.user.username;
        let farm_name = req.params.farm_name;
        let field_name = req.params.field_name;
        let mission_date = req.params.mission_date;
        let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets", farm_name, field_name, mission_date);

        if (!(fpath_exists(image_set_dir))) {
            return res.redirect(process.env.AC_PATH);
        }

        let image_set_key = ([username, farm_name, field_name, mission_date]).join("/");

        for (let socket_id of Object.keys(socket_api.workspace_id_to_key)) {
            if (socket_api.workspace_id_to_key[socket_id] === image_set_key) {
                console.log("The workspace is in use", image_set_key);
                return res.redirect(process.env.AC_PATH + "/home/" + username); 
            }
        }


        let maintenance_time = "";
        let maintenance_path = path.join(USR_SHARED_ROOT, "maintenance.json");
        if (fpath_exists(maintenance_path)) {
            try {
                maintenance_log = JSON.parse(fs.readFileSync(maintenance_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            maintenance_time = maintenance_log["maintenance_time"];
        }


        glob(path.join(image_set_dir, "images", "*"), function(error, image_paths) {
            if (error) {
                return res.redirect(process.env.AC_PATH);
            }
            let image_ext = image_paths[0].substring(image_paths[0].length - 4);

            let overlay_appearance;
            let overlay_appearance_path = path.join(USR_DATA_ROOT, username, "overlay_appearance.json");
            try {
                overlay_appearance = JSON.parse(fs.readFileSync(overlay_appearance_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            let hotkeys;
            let hotkeys_path = path.join(USR_DATA_ROOT, username, "hotkeys.json");
            try {
                hotkeys = JSON.parse(fs.readFileSync(hotkeys_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            console.log("getting annotations");
            let annotations_dir = path.join(image_set_dir, "annotations");
            let annotations_path = path.join(annotations_dir, "annotations.json");
            let annotations;
            try {
                annotations = JSON.parse(fs.readFileSync(annotations_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            console.log("getting metadata");
            let metadata_path = path.join(image_set_dir, "metadata", "metadata.json");
            let metadata;
            try {
                metadata = JSON.parse(fs.readFileSync(metadata_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            console.log("getting camera specs");
            let camera_specs_path = path.join(USR_DATA_ROOT, username, "cameras", "cameras.json");
            let camera_specs;
            try {
                camera_specs = JSON.parse(fs.readFileSync(camera_specs_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            let excess_green_record;
            console.log("getting exg record");
            let excess_green_record_path = path.join(image_set_dir, "excess_green", "record.json");
            
            try {
                excess_green_record = JSON.parse(fs.readFileSync(excess_green_record_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            let tags;
            console.log("getting tags");
            let tags_path = path.join(image_set_dir, "annotations", "tags.json");
            
            try {
                tags = JSON.parse(fs.readFileSync(tags_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }


            console.log("getting dzi image paths");
            let dzi_images_dir = path.join(image_set_dir, "dzi_images");
            let dzi_image_paths = [];
            for (let image_name of Object.keys(annotations)) {
                let dzi_image_path = path.join(process.env.AC_PATH, dzi_images_dir, image_name + ".dzi");
                dzi_image_paths.push(dzi_image_path);
            }

            let image_set_info = {
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "image_ext": image_ext,
            }
    
            let data = {};

            data["cur_page"] = "workspace";
            data["image_set_info"] = image_set_info;
            data["metadata"] = metadata;
            data["dzi_image_paths"] = nat_orderBy.orderBy(dzi_image_paths);
            data["annotations"] = annotations;
            data["excess_green_record"] = excess_green_record;
            data["tags"] = tags;
            data["camera_specs"] = camera_specs;
            data["overlay_appearance"] = overlay_appearance;
            data["hotkeys"] = hotkeys;
            data["maintenance_time"] = maintenance_time;

            res.render("workspace", {username: username, data: data});

        });
    }
    else {
        return res.redirect(process.env.AC_PATH);
    }
}

function notify_scheduler(request) {

    let response = {};

    return new Promise((resolve, reject) => {

        let data = JSON.stringify(request);

        let options = {
            hostname: process.env.AC_IP,
            port: parseInt(process.env.AC_PY_PORT),
            path: process.env.AC_PATH + '/add_request',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': data.length
            }
        };

        let req = http.request(options, res => {
            console.log(`statusCode: ${res.statusCode}`);


            let chunks = [];

            res.on("data", function(chunk) {
                chunks.push(chunk);
            });

            res.on("end", function() {
                let body = Buffer.concat(chunks);
                let result = JSON.parse(body);
                console.log("Got result", result);
                if (!("message" in result)) {
                    response.error = true;
                    response.message = "Got an unexpected response from the job scheduler.";
                    resolve(response);
                }
                if (result["message"] === "ok") {
                    response.error = false;
                    response.message = "The job was successfully enqueued.";
                    resolve(response);
                }
                else {
                    response.error = true;
                    response.message = result["message"];
                    resolve(response);
                }
            });
        });

        req.on("error", error => {
            console.log(error);
            response.error = true;
            response.message = "Failed to contact the job scheduler.";
            resolve(response);
        });

        req.write(data);
        req.end();
    });
}

function results_name_is_valid(results_name) {

    let format = /[`!@#$%^&*()+\=\[\]{};':"\\|,<>\/?~]/;
    if (format.test(results_name)) {
        return false;
    }
    if ((results_name.length < 1) || (results_name.length > 50)) {
        return false;
    }
    return true;
}


function results_comment_is_valid(results_comment) {

    let format = /[`!@#$%^&*\=\[\]{}|<>?~]/;
    if (format.test(results_comment)) {
        return false;
    }
    if (results_comment.length > 255) {
        return false;
    }
    return true;
}

function get_models(model_paths, username, farm_name, field_name, mission_date, object_classes) {

    let models = [];
    for (let model_path of model_paths) {
        let log_path = path.join(model_path, "log.json");
        let log = JSON.parse(fs.readFileSync(log_path, 'utf8'));

        let image_set_used_to_train_model = false;
        for (let image_set of log["image_sets"]) {
            if ((image_set["username"] === username && image_set["farm_name"] === farm_name) &&
                (image_set["field_name"] === field_name && image_set["mission_date"] === mission_date)) {
                    image_set_used_to_train_model = true;
                    break;
            }
        }

        let valid = true;
        if (object_classes.length != log["object_classes"].length) {
            valid = false;
        }
        if (valid) {
            for (let i = 0; i < object_classes.length; i++) {
                if (object_classes[i] !== log["object_classes"][i]) {
                    valid = false;
                }
            }
        }
        if (valid) {
            models.push({
                "model_creator": log["model_creator"],
                "model_name": log["model_name"],
                "image_set_used_to_train_model": image_set_used_to_train_model
            });
        }
    }
    return models;
}



exports.post_annotations_upload = function(req, res, next) {
    let response = {};

    let username = req.session.user.username;
    let farm_name = req.params.farm_name;
    let field_name = req.params.field_name;
    let mission_date = req.params.mission_date;

    let annotations_file = req.files[0].buffer;

    let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets", farm_name,
                                        field_name, mission_date);

    console.log("image_set_dir", image_set_dir);

    console.log("trying to parse annotations_file")
    let annotations;
    try {
        annotations = JSON.parse(annotations_file);
    }
    catch (error) {
        return res.status(422).json({
            error: "Unable to parse the annotations file."
        });
    }


    let metadata_path = path.join(image_set_dir, "metadata", "metadata.json");
    let metadata;
    try {
        metadata = JSON.parse(fs.readFileSync(metadata_path, 'utf8'));
    }
    catch(error) {
        return res.status(422).json({
            error: "Failed to read metadata file."
        });
    }

    let annotations_path = path.join(image_set_dir, "annotations", "annotations.json");
    let existing_annotations;
    try {
        existing_annotations = JSON.parse(fs.readFileSync(annotations_path, 'utf8'));
    }
    catch(error) {
        return res.status(422).json({
            error: "Failed to read existing annotations."
        });
    }




    let min_box_dim = 1;
    let max_box_dim = 800;
    let new_annotations;


    new_annotations = existing_annotations;

    for (let entry_name of Object.keys(annotations)) {

        let image_name;
        let image_path;
        let entry_pieces = entry_name.split(".");
        image_name = entry_pieces[0];
        if (entry_pieces.length == 1) {
            image_path = path.join(image_set_dir, "images", image_name + ".*");
        }
        else if (entry_pieces.length == 2) {
            image_path = path.join(image_set_dir, "images", entry_name);
        }
        else {
            return res.status(422).json({
                error: "The uploaded annotation file contains an invalid key: " + entry_name
            });
        }

        let matched_image_paths;
        try {
            matched_image_paths = glob.sync(image_path);
        }
        catch (error) {
            return res.status(422).json({
                error: "An error occurred while checking the validity of the annotation file's keys."
            });
        }
        console.log("image_paths", matched_image_paths);
        if (matched_image_paths.length != 1) {
            return res.status(422).json({
                error: "The uploaded annotation file contains an invalid key: " + entry_name
            });
        }

        new_annotations[image_name] = {
            "boxes": [],
            "classes": [],
            "regions_of_interest": [],
            "fine_tuning_regions": [],
            "test_regions": [],
            "source": "uploaded"
        };

        for (let key of ["regions_of_interest", "fine_tuning_regions", "test_regions"])
            if (key in annotations[entry_name]) {
                if (!(Array.isArray(annotations[entry_name][key]))) {
                    return res.status(422).json({
                        error: "The uploaded annotations file contains an invalid value (not an array). Problematic key: " + entry_name + "."
                    });
                }
                let num_regions = annotations[entry_name][key].length;
                if (num_regions > 99) {
                    return res.status(422).json({
                        error: "The uploaded annotations file contains too many regions" +
                                " for image " + entry_name + ". A maximum of 99 of each type of region " + 
                                " is allowed per image. " + num_regions + " were provided."
                    });
                }

                for (let i = 0; i < annotations[entry_name][key].length; i++) {
                    let poly = annotations[image_name][key][i];
                    
                    if (!(Array.isArray(poly))) {
                        return res.status(422).json({
                            error: "The uploaded annotations file contains an invalid polygon (not an array). Problematic key: " + entry_name + "."
                        });
                    }
                    if (poly.length < 3) {
                        return res.status(422).json({
                            error: "The uploaded annotations file contains an invalid polygon (number of points is less than 3). Problematic key: " + entry_name + "."
                        });
                    }

                    let image_w = metadata["images"][image_name]["width_px"];
                    let image_h = metadata["images"][image_name]["height_px"];

                    let uploaded_poly = [];
                    for (let pt of poly) {
                        if (!(Array.isArray(pt))) {
                            return res.status(422).json({
                                error: "The uploaded annotations file contains an invalid polygon (at least one point is not an array). Problematic key: " + entry_name + "."
                            });
                        }
                        if (pt.length != 2) {
                            return res.status(422).json({
                                error: "The uploaded annotations file contains an invalid polygon (at least one point is not an array of length 2). Problematic key: " + entry_name + "."
                            });
                        }

                        if ((pt[0] < 0 || pt[0] > image_w) || (pt[1] < 0 || pt[1] > image_h)) {
                            return res.status(422).json({
                                error: "The uploaded annotations file contains a polygon with a point located outside of the image boundaries. Problematic key: " + entry_name + "."
                            });
                        }

                        let all_numbers = pt.every(element => { return typeof element === "number"; });
                        if (!(all_numbers)) {
                            return res.status(422).json({
                                error: "The uploaded annotations file contains a polygon coordinate with non-numeric values. Problematic key: " + entry_name + "."
                            });
                        }


                        uploaded_poly.push([pt[1], pt[0]]);
                    }

                    new_annotations[image_name][key].push(
                        uploaded_poly
                    );
                }
            }

        let annotation_lst_length = 0;
        let class_lst_length = 0;
        if ("classes" in annotations[entry_name]) {
            class_lst_length = annotations[entry_name]["classes"].length;
        }
        if ("boxes" in annotations[entry_name]) {
            annotation_lst_length = annotations[entry_name]["boxes"].length;
        }
        if (annotation_lst_length != class_lst_length) {
            return res.status(422).json({
                error: "The length of the box annotation list must match the length of the class list in all images. Problematic key: " + entry_name + "."
            });
        }
        if ("classes" in annotations[entry_name]) {

            if (!(Array.isArray(annotations[entry_name]["classes"]))) {
                return res.status(422).json({
                    error: "The uploaded annotations file contains an invalid value (not an array). Problematic key: " + entry_name + "."
                });
            }
            new_annotations[image_name]["classes"] = [];
            for (let cls_ind of annotations[entry_name]["classes"]) {
                if (cls_ind < 0) {
                    return res.status(422).json({
                        error: "Invalid class value detected. Class values must be non-negative integers. Problematic key: " + entry_name + "."
                    });
                }
                if (cls_ind >= metadata["object_classes"].length) {
                    return res.status(422).json({
                        error: "Invalid class value detected. Problematic key: " + entry_name + "."
                    });
                }

                new_annotations[image_name]["classes"].push(cls_ind);
            }
        }


        if ("boxes" in annotations[entry_name])  {
            if (!(Array.isArray(annotations[entry_name]["boxes"]))) {
                return res.status(422).json({
                    error: "The uploaded annotations file contains an invalid value (not an array). Problematic key: " + entry_name + "."
                });
            }

            for (let i = 0; i < annotations[entry_name]["boxes"].length; i++) {
                let box = annotations[entry_name]["boxes"][i];
                if (!(Array.isArray(box))) {
                    return res.status(422).json({
                        error: "The uploaded annotations file contains an invalid box (not an array). Problematic key: " + entry_name + "."
                    });
                }
                if (box.length != 4) {
                    return res.status(422).json({
                        error: "The uploaded annotations file contains a malformed box (number of elements is not equal to 4). Problematic key: " + entry_name + "."
                    });
                }

                let all_numbers = box.every(element => { return typeof element === "number"; });
                if (!(all_numbers)) {
                    return res.status(422).json({
                        error: "The uploaded annotations file contains a box with non-numeric values. Problematic key: " + entry_name + "."
                    });
                }

                let y_min = Math.round(box[1]);
                let x_min = Math.round(box[0]);
                let y_max = Math.round(box[3]);
                let x_max = Math.round(box[2]);

                let image_w = metadata["images"][image_name]["width_px"];
                let image_h = metadata["images"][image_name]["height_px"];

                if (((y_min < 0) || (x_min < 0)) || ((y_max > image_h) || (x_max > image_w))) {
                    return res.status(422).json({
                        error: "The uploaded annotations file contains a box with coordinates outside of the image boundaries. Problematic key: " + entry_name + "."
                    });
                }

                let box_height = y_max - y_min;
                let box_width = x_max - x_min;

                let pixel_text;
                if (box_height == 1) {
                    pixel_text = "pixel";
                }
                else {
                    pixel_text = "pixels";
                }

                if (box_height < min_box_dim) {

                    return res.status(422).json({
                        error: "At least one uploaded box has a height that is smaller than the minimum allowed height. " +
                                "(Box height: " + box_height + " " + pixel_text + ". Minimum allowed height: " + min_box_dim + 
                                " pixel. Problematic key: " + entry_name + ".)"
                    });
                }
                
                if (box_width < min_box_dim) {
                    return res.status(422).json({
                        error: "At least one uploaded box has a width that is smaller than the minimum allowed width. " +
                            "(Box width: " + box_width + " " + pixel_text + ". Minimum allowed width: " + min_box_dim + 
                            " pixel. Problematic key: " + entry_name + ".)"
                    });
                }
                // if (annotation_key == "annotations") {
                //     if (box_height > max_box_dim) {
                //         return res.status(422).json({
                //             error: "At least one uploaded box has a height that is larger than the maximum allowed height. " +
                //                     "(Box height: " + box_height + " " + pixel_text + ". Maximum allowed height: " + max_box_dim + 
                //                     " pixels. Problematic key: " + entry_name + ".)"
                //         });
                //         // response.message = "At least one uploaded box has a height that is larger than the maximum allowed height. " +
                //         //                     "(Box height: " + box_height + ". Maximum allowed height: " + max_box_dim + ".)";
                //         // response.error = true;
                //         // return res.json(response);
                //     }
                //     if (box_width > max_box_dim) {
                //         return res.status(422).json({
                //             error: "At least one uploaded box has a width that is larger than the maximum allowed width. " +
                //                 "(Box width: " + box_width + " " + pixel_text + ". Maximum allowed width: " + max_box_dim + 
                //                 " pixels. Problematic key: " + entry_name + ".)"
                //         });
                //         // response.message = "At least one uploaded box has a width that is larger than the maximum allowed width. " +
                //         //                     "(Box width: " + box_width + ". Maximum allowed width: " + max_box_dim + ".)";
                //         // response.error = true;
                //         // return res.json(response);
                //     }
                // }

                new_annotations[image_name]["boxes"].push([
                    y_min, x_min, y_max, x_max
                ]);
            }
        }
    }

    try {
        fs.writeFileSync(annotations_path, JSON.stringify(new_annotations));
    }
    catch (error) {
        return res.status(422).json({
            error: "Failed to save uploaded annotations."
        });
    }


    let empty = true;
            
    for (let image_name of Object.keys(new_annotations)) {
        if (new_annotations[image_name]["fine_tuning_regions"].length > 0) {
            empty = false;
            break;
        }
        if (new_annotations[image_name]["test_regions"].length > 0) {
            empty = false;
            break;
        }
    }

    let image_set_key = ([username, farm_name, field_name, mission_date]).join("/");

    let image_sets_path;
    if (metadata["is_public"]) {
        image_sets_path = path.join(USR_SHARED_ROOT, "public_image_sets.json");
    }
    else {
        image_sets_path = path.join(USR_DATA_ROOT, username, "private_image_sets.json");
    }
    image_sets_mutex.acquire()
    .then(function(release) {
        let image_sets;
        try {
            image_sets = JSON.parse(fs.readFileSync(image_sets_path, 'utf8'));
        }
        catch (error) {
            release();
            response.error = true;
            response.message = "Failed to read image sets file.";
            return res.json(response);
        }

        if (empty) {
            delete image_sets[image_set_key];
        }
        else {
            image_sets[image_set_key] = {
                "object_classes": metadata["object_classes"]
            };
        }

        try {
            fs.writeFileSync(image_sets_path, JSON.stringify(image_sets));
        }
        catch (error) {
            release();
            response.message = "Failed to write image sets file.";
            response.error = true;
            return res.json(response);
        }


        release();

        response.error = false;
        response.annotations = new_annotations;
    
        return res.json(response);

    }).catch(function(error) {
        console.log(error);
        response.error = true;
        response.message = "Failed to acquire image sets mutex.";
        return res.json(response);
    });

}


function verify_annotations(annotations) {
    // TODO
}


function verify_excess_green_record(excess_green_record) {
    // TODO
}

function verify_tags(tags) {
    // TODO
}

exports.post_workspace = async function(req, res, next) {

    let response = {};

    let username = req.session.user.username;
    let farm_name = req.params.farm_name;
    let field_name = req.params.field_name;
    let mission_date = req.params.mission_date;
    let action = req.body.action;

    let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets", farm_name,
                                        field_name, mission_date);

    if (action === "save_annotations") {
        console.log("saving annotations");

        let annotations_path = path.join(image_set_dir, "annotations", "annotations.json");
        let annotations_backup_path = path.join(image_set_dir, "annotations", "backup_annotations.json");
        let annotations = JSON.parse(req.body.annotations);
        let excess_green_record = JSON.parse(req.body.excess_green_record);
        let tags = JSON.parse(req.body.tags);

        verify_annotations(annotations);
        verify_excess_green_record(excess_green_record);
        verify_tags(tags);


        fs.rename(annotations_path, annotations_backup_path, (error) => {
            if (error) {
                response.message = "Failed to write backup annotations file.";
                response.error = true;
                return res.json(response);
            }
        
            try {
                fs.writeFileSync(annotations_path, req.body.annotations);
            }
            catch (error) {
                response.message = "Failed to write annotations.";
                response.error = true;
                return res.json(response);
            }

            let excess_green_record_path = path.join(image_set_dir, "excess_green", "record.json");
            try {
                fs.writeFileSync(excess_green_record_path, req.body.excess_green_record);
            }
            catch (error) {
                response.error = true;
                response.message = "Failed to write excess green record.";
                return res.json(response);
            }

            let tags_path = path.join(image_set_dir, "annotations", "tags.json");
            try {
                fs.writeFileSync(tags_path, req.body.tags);
            }
            catch (error) {
                response.error = true;
                response.message = "Failed to write tags.";
                return res.json(response);
            }

            let empty = true;
            
            for (let image_name of Object.keys(annotations)) {
                if (annotations[image_name]["fine_tuning_regions"].length > 0) {
                    empty = false;
                    break;
                }
                if (annotations[image_name]["test_regions"].length > 0) {
                    empty = false;
                    break;
                }
            }

            let image_sets_path;
            if (JSON.parse(req.body.is_public)) {
                image_sets_path = path.join(USR_SHARED_ROOT, "public_image_sets.json");
            }
            else {
                image_sets_path = path.join(USR_DATA_ROOT, username, "private_image_sets.json");
            }

            let image_set_key = ([username, farm_name, field_name, mission_date]).join("/");
            

            image_sets_mutex.acquire()
            .then(function(release) {
                let image_sets;
                try {
                    image_sets = JSON.parse(fs.readFileSync(image_sets_path, 'utf8'));
                }
                catch (error) {
                    release();
                    response.error = true;
                    response.message = "Failed to read image sets file.";
                    return res.json(response);
                }

                if (empty) {
                    delete image_sets[image_set_key];
                }
                else {
                    image_sets[image_set_key] = {
                        "object_classes": req.body.object_classes.split(",")
                    };
                }

                try {
                    fs.writeFileSync(image_sets_path, JSON.stringify(image_sets));
                }
                catch (error) {
                    release();
                    response.message = "Failed to write image sets file.";
                    response.error = true;
                    return res.json(response);
                }


                release();

                response.error = false;
                return res.json(response);

            }).catch(function(error) {
                console.log(error);
                response.error = true;
                response.message = "Failed to acquire image sets mutex.";
                return res.json(response);
            });
        });

    }
    else if (action === "download_annotations") {

        let annotations_path = path.join(image_set_dir, "annotations", "annotations.json");
        let annotations;
        try {
            annotations = JSON.parse(fs.readFileSync(annotations_path, 'utf8'));
        }
        catch(error) {
            response.error = true;
            response.message = "Failed to read annotations file.";
            return res.json(response);
        }

        let download_annotations = {};
        for (let image_name of Object.keys(annotations)) {
            download_annotations[image_name] = {};


            for (let annotation_key of ["regions_of_interest", "fine_tuning_regions", "test_regions"]) {
                download_annotations[image_name][annotation_key] = [];
                for (let i = 0; i < annotations[image_name][annotation_key].length; i++) {
                    let download_region = [];
                    for (let j = 0; j < annotations[image_name][annotation_key][i].length; j++) {
                        let pt = annotations[image_name][annotation_key][i][j];
                        let download_pt = [
                            pt[1], pt[0]
                        ];
                        download_region.push(download_pt);
                    }
                    download_annotations[image_name][annotation_key].push(download_region);
                }

            }

            download_annotations[image_name]["classes"] = annotations[image_name]["classes"];

            download_annotations[image_name]["boxes"] = []

            for (let i = 0; i < annotations[image_name]["boxes"].length; i++) {
                let box = annotations[image_name]["boxes"][i];
                let download_box = [
                    box[1], box[0], box[3], box[2]
                ];

                download_annotations[image_name]["boxes"].push(download_box);
            }

        }

        let annotations_download_path = path.join(image_set_dir, "annotations", "download_annotations.json");
        try {
            fs.writeFileSync(annotations_download_path, JSON.stringify(download_annotations));
        }
        catch (error) {
            response.error = true;
            response.message = "Failed to write downloadable annotations file.";
            return res.json(response);
        }

        response.error = false;
        return res.json(response);

    }
    else if (action === "build_map") {

        let prediction_dir = path.join(image_set_dir, "model", "prediction");
        let maps_dir = path.join(image_set_dir, "maps");
        let rebuild_command = "python ../../backend/src/interpolate.py " + username + " " +
            farm_name + " " + field_name + " " + mission_date + " " + prediction_dir + 
            " " + maps_dir + " " + req.body.class_index + " obj_density";

        if (req.body.interpolation === "nearest") {
            rebuild_command = rebuild_command + " -nearest";
        }
        if (req.body.tile_size !== "") {
            rebuild_command = rebuild_command + " -tile_size " + req.body.tile_size;
        }
        console.log(rebuild_command);
        let result = exec(rebuild_command, {shell: "/bin/bash"}, function (error, stdout, stderr) {
            if (error) {
                console.log(error.stack);
                console.log('Error code: '+error.code);
                console.log('Signal received: '+error.signal);
                response.error = true;
            }
            else {
                response.error = false;
            }
            return res.json(response);
        });
    }
    else if (action === "fetch_models") {
        let object_classes = req.body.object_classes.split(",");

        let models = [];
        let usr_dirs;
        try {
           usr_dirs = get_subdirpaths(USR_DATA_ROOT);
        }
        catch (error) {
            response.message = "Failed to retrieve models.";
            response.error = true;
            return res.json(response);
        }

        for (let usr_dir of usr_dirs) {
            let public_models_dir = path.join(usr_dir, "models", "available", "public");

            let public_dirs;
            try {
                public_dirs = get_subdirpaths(public_models_dir);
            }
            catch (error) {
                response.message = "Failed to retrieve models.";
                response.error = true;
                return res.json(response);
            }

            let cur_models;
            try {
                cur_models = get_models(public_dirs, username, farm_name, field_name, mission_date, object_classes);
            }
            catch (error) {
                response.message = "Failed to retrieve models.";
                response.error = true;
                return res.json(response);
            }

            models = models.concat(cur_models);

            if (path.basename(usr_dir) === username) {

                let private_models_dir = path.join(usr_dir, "models", "available", "private");

                let private_dirs;
                try {
                    private_dirs = get_subdirpaths(private_models_dir);
                }
                catch (error) {
                    response.message = "Failed to retrieve models.";
                    response.error = true;
                    return res.json(response);
                }

                console.log("private_dirs", private_dirs)

                try {
                    cur_models = get_models(private_dirs, username, farm_name, field_name, mission_date, object_classes);
                }
                catch (error) {
                    response.message = "Failed to retrieve models.";
                    response.error = true;
                    return res.json(response);
                }

                models = models.concat(cur_models);
            }

        }

        response.error = false;
        response.models = nat_orderBy.orderBy(models, 
            [v => v.model_creator, v => v.model_name], 
            ['asc', 'asc']);
        return res.json(response);
    }
    else if (action === "inspect_model") {
        let model_creator = req.body.model_creator;
        let model_name = req.body.model_name;

        let public_log_path = path.join(USR_DATA_ROOT, model_creator, "models", "available", "public", model_name, "log.json");
        let private_log_path = path.join(USR_DATA_ROOT, model_creator, "models", "available", "private", model_name, "log.json");
        let log;
        try {
            log = JSON.parse(fs.readFileSync(public_log_path, 'utf8'));
        }
        catch(error) {
            try {
                log = JSON.parse(fs.readFileSync(private_log_path, 'utf8'));
            }
            catch(error) {
                response.error = true;
                return res.json(response);
            }
        }

        response.model_log = log;
        response.error = false;
        return res.json(response);
    }
    else if (action === "fetch_model_annotations") {
        let model_creator = req.body.model_creator;
        let model_name = req.body.model_name;
        let image_set_username = req.body.username;
        let image_set_farm_name = req.body.farm_name;
        let image_set_field_name = req.body.field_name;
        let image_set_mission_date = req.body.mission_date;



        let public_model_dir = path.join(USR_DATA_ROOT, model_creator, "models", "available", "public", model_name);
        let private_model_dir = path.join(USR_DATA_ROOT, model_creator, "models", "available", "private", model_name);

        let public_annotations_path = path.join(public_model_dir, "annotations", image_set_username,
                                            image_set_farm_name, image_set_field_name, image_set_mission_date,
                                            "annotations.json");

        let private_annotations_path = path.join(private_model_dir, "annotations", image_set_username,
                                            image_set_farm_name, image_set_field_name, image_set_mission_date,
                                            "annotations.json");
        let annotations;
        try {
            annotations = JSON.parse(fs.readFileSync(public_annotations_path, 'utf8'));
        }
        catch (error) {
            try {
                annotations = JSON.parse(fs.readFileSync(private_annotations_path, 'utf8'));
            }
            catch (error) {
                response.message = "Failed to retrieve image set annotations."
                response.error = true;
                return res.json(response);
            }
        }

        response.annotations = annotations;
        response.error = false;
        return res.json(response);
    }
    else if (action === "fine_tune") {

        console.log("Fine-tuning requested");

        let training_regime = req.body.training_regime;

        let job_key = ([username, farm_name, field_name, mission_date]).join("/");
        let request = {
            "key": job_key,
            "task": "fine_tune",
            "training_regime": training_regime,
            "request_time": Math.floor(Date.now() / 1000),
            "username": username,
            "farm_name": farm_name,
            "field_name": field_name,
            "mission_date": mission_date
        };

        if (training_regime === "fixed_num_epochs") {
            request["num_epochs"] = parseInt(req.body.num_epochs);
        }
        else {
            request["training_percent"] = parseFloat(req.body.training_percent);
            request["improvement_tolerance"] = parseFloat(req.body.improvement_tolerance);
        }

        response = await notify_scheduler(request); 
        return res.json(response);

    }
    else if (action === "predict") {

        console.log("Prediction requested");

        let image_names = JSON.parse(req.body.image_names);
        let regions = JSON.parse(req.body.regions);
        let save_result = JSON.parse(req.body.save_result);
        let regions_only = JSON.parse(req.body.regions_only);
        let calculate_vegetation_coverage = JSON.parse(req.body.calculate_vegetation_coverage);
        let job_key = ([username, farm_name, field_name, mission_date]).join("/");

        let request = {
            "key": job_key,
            "task": "predict",
            "request_time": Math.floor(Date.now() / 1000),
            "username": username,
            "farm_name": farm_name,
            "field_name": field_name,
            "mission_date": mission_date,
            "image_names": image_names,
            "regions": regions,
            "save_result": save_result,
            "regions_only": regions_only,
            "calculate_vegetation_coverage": calculate_vegetation_coverage
        };

        if (save_result) {
            let results_name = req.body.results_name;
            if (!(results_name_is_valid(results_name))) {
                response.message = "Results name is invalid";
                response.error = true;
                return res.json(response);
            }

            let results_comment = req.body.results_comment;
            if (!(results_comment_is_valid(results_comment))) {
                response.message = "Results comment is invalid";
                response.error = true;
                return res.json(response);
            }

            request["result_uuid"] = uuidv4().toString();
            request["results_name"] = results_name;
            request["results_comment"] = results_comment;
        }

        response = await notify_scheduler(request);
        return res.json(response);
    }
    else if (action === "retrieve_predictions") {

        let image_names = req.body.image_names.split(",");
        response.predictions = {};
        for (let image_name of image_names) {
            let prediction_path = path.join(
                image_set_dir, 
                "model", 
                "prediction",
                image_name + ".json"
            );

            if (fs.existsSync(prediction_path)) {

                let image_predictions;
                try {
                    image_predictions = JSON.parse(fs.readFileSync(prediction_path, 'utf8'));
                }
                catch (error) {
                    console.log(error);
                    response.error = true;
                    response.message = "Failed to retrieve predictions.";
                    return res.json(response);
                }
                response.predictions[image_name] = image_predictions;
            }
        }
        response.error = false;
        return res.json(response);

    }
    else if (action === "switch_model") {

        let job_key = ([username, farm_name, field_name, mission_date]).join("/");
        let request = {
            "key": job_key,
            "task": "switch",
            "request_time": Math.floor(Date.now() / 1000),
            "username": username,
            "farm_name": farm_name,
            "field_name": field_name,
            "mission_date": mission_date,
            "model_name": req.body.model_name,
            "model_creator": req.body.model_creator,
        };

        response = await notify_scheduler(request);
        return res.json(response);

    }
}


function isNumeric(str) {
    if (typeof str != "string") return false;
    return !isNaN(str) &&
           !isNaN(parseFloat(str));
}

function remove_image_set(username, farm_name, field_name, mission_date) {

    console.log("remove_image_set");
    console.log("username", username);
    console.log("farm_name", farm_name);
    console.log("field_name", field_name);
    console.log("mission_date", mission_date);

    if ((username === "" || farm_name === "") || (field_name === "" || mission_date === "")) {
        throw "Empty string argument provided";
    }

    if ((username == null || farm_name == null) || (field_name == null || mission_date == null)) {
        throw "Null argument provided";
    }

    let farm_dir = path.join(USR_DATA_ROOT, username, "image_sets", farm_name);
    let field_dir = path.join(farm_dir, field_name);
    let mission_dir = path.join(field_dir, mission_date);

    if (fs.existsSync(mission_dir)) {
        console.log("removing mission_dir", mission_dir);
        fs.rmSync(mission_dir, { recursive: true, force: false });
    }
    if (fs.existsSync(field_dir)) {
        let missions = get_subdirnames(field_dir);
        if (missions.length == 0) {
            console.log("removing field_dir", field_dir);
            fs.rmSync(field_dir, { recursive: true, force: false });
        }
    }
    if (fs.existsSync(farm_dir)) {
        let fields = get_subdirnames(farm_dir);
        if (fields.length == 0) {
            console.log("removing farm_dir", farm_dir);
            fs.rmSync(farm_dir, { recursive: true, force: false });
        }
    }
}

exports.post_orthomosaic_upload = function(req, res, next) {

    let dzchunkindex = req.body.dzchunkindex;
    let dztotalchunkcount = req.body.dztotalchunkcount;

    let upload_uuid = req.body.upload_uuid;
    let username = req.session.user.username;
    let farm_name = req.body.farm_name;
    let field_name = req.body.field_name;
    let mission_date = req.body.mission_date;
    let object_classes = req.body.object_classes.split(",");
    let is_public = JSON.parse(req.body.is_public);
    let camera_height = req.body.camera_height;
    let queued_filenames = req.body.queued_filenames.split(",");

    let file = req.files[0];
    let filename = file.originalname;

    let first = dzchunkindex == 0;
    let last = dzchunkindex == dztotalchunkcount - 1;


    let image_sets_root = path.join(USR_DATA_ROOT, username, "image_sets");
    let farm_dir = path.join(image_sets_root, farm_name);
    let field_dir = path.join(farm_dir, field_name);
    let mission_dir = path.join(field_dir, mission_date);
    let images_dir = path.join(mission_dir, "images");
    let fpath = path.join(images_dir, filename);
    

    if (queued_filenames.length != 1) {
        return res.status(422).json({
            error: "Only one orthomosaic can be uploaded at a time."
        });

    }

    console.log("dzchunkindex", dzchunkindex);
    console.log("dzchunktotalcount", dztotalchunkcount);
    console.log("first?", first);
    console.log("last?", last);
    console.log("upload_uuid", upload_uuid);

    if (first) {
        if (upload_uuid in active_uploads) {
            return res.status(422).json({
                error: "Upload key conflict."
            });
        }
        else {
            active_uploads[upload_uuid] = {
                "status": "active"
            };
        }
    }
    else {
        if (!(upload_uuid in active_uploads)) {
            return res.status(422).json({
                error: "Upload is no longer active."
            });
        }
    }


    if (first) {
        let split_filename = filename.split(".");

        if (FILE_FORMAT.test(filename)) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "The provided filename contains illegal characters."
            });
        }

        
        if ((split_filename.length != 1) && (split_filename.length != 2)) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "The provided filename contains an illegal '.' character."
            });
        }

        let extensionless_fname = split_filename[0];
        if (extensionless_fname.length > MAX_EXTENSIONLESS_FILENAME_LENGTH) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "The provided filename exceeds the maximum allowed length of " + MAX_EXTENSIONLESS_FILENAME_LENGTH + " characters."
            });
        }


        if (fpath_exists(mission_dir)) {
            
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "The provided farm-field-mission combination already exists."
            });
        }

        console.log("checking components");
        let id_components = [farm_name, field_name, mission_date];
        for (let id_component of id_components) {
            if (id_component.length < 3) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "The provided farm name, field name, or mission date is too short."
                });
            }
            if (id_component.length > 20) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "The provided farm name, field name, or mission date is too long."
                });
            }
            if (FARM_FIELD_MISSION_FORMAT.test(id_component)) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "The provided farm name, field name, or mission date contains illegal characters."
                });
            }
        }
        let date = new Date(mission_date);
        if (!(date.isValid())) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "The provided mission date is invalid."
            });
        }

        let objects_path = path.join(USR_SHARED_ROOT, "objects.json");
        let objects;
        try {
            objects = JSON.parse(fs.readFileSync(objects_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.status(422).json({
                error: "Failed to read objects file."
            });
        }
        for (let object_class of object_classes) {
            if (!(objects["object_names"].includes(object_class))) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "One of the provided object classes ('" + object_class + "') is not recognized by the system."
                });
            }
        }

        if (camera_height.length > 0) {
            if (isNumeric(camera_height)) {
                let numeric_camera_height = parseFloat(camera_height);
                if (numeric_camera_height < MIN_CAMERA_HEIGHT || numeric_camera_height > MAX_CAMERA_HEIGHT) {
                    delete active_uploads[upload_uuid];
                    return res.status(422).json({
                        error: "The provided camera height falls outside of the accepted range."
                    });
                }
            }
            else {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "The provided camera height is invalid."
                });
            }
        }


        console.log("Making the images directory");
        fs.mkdirSync(images_dir, { recursive: true });
    }
    else {
        if (!(fpath_exists(mission_dir))) {
            try {
                remove_image_set(username, farm_name, field_name, mission_date);
            }
            catch (error) {
                console.log("Failed to remove image set");
                console.log(error);
            }
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "Image set directories were not created by initial request."
            });
        }
    }

    if (first) {
        let writeStream = fs.createWriteStream(fpath);
        active_uploads[upload_uuid]["stream"] = writeStream;
    }


    active_uploads[upload_uuid]["stream"].write(file.buffer, file.encoding, function(error) {
        if (error) {
            try {
                remove_image_set(username, farm_name, field_name, mission_date);
            }
            catch (error) {
                console.log("Failed to remove image set");
                console.log(error);
            }
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "Error occurred when writing image file."
            });
        }

        if (last) {
            active_uploads[upload_uuid]["stream"].end();

            let config = {
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "object_classes": object_classes,
                "is_public": is_public,
                "is_ortho": true
                
            }
            if (camera_height.length > 0) {
                config["camera_height"] = parseFloat(camera_height);
            }
            else {
                config["camera_height"] = "";
            }

            let config_path = path.join(mission_dir, "config.json");
            try {
                fs.writeFileSync(config_path, JSON.stringify(config));
            }
            catch (error) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "Error occurred when writing configuration file."
                });
            }


            let process_upload_command = "python ../../backend/src/process_upload.py " + mission_dir;
            exec(process_upload_command, {shell: "/bin/bash"}, function (error, stdout, stderr) {
                if (error) {
                    console.log(error.stack);
                    console.log('Error code: '+error.code);
                    console.log('Signal received: '+error.signal);
                }
            });

            delete active_uploads[upload_uuid];

        }

        return res.sendStatus(200);
    });
}


exports.post_image_set_upload = async function(req, res, next) {

    let upload_uuid;
    let farm_name;
    let field_name;
    let mission_date;
    let object_classes;
    let is_public;
    let first;
    let last;
    let queued_filenames;
    let camera_height;

    let username = req.session.user.username;
    
    if (req.files.length > 1) {
        upload_uuid = req.body.upload_uuid[0];
        farm_name = req.body.farm_name[0];
        field_name = req.body.field_name[0];
        mission_date = req.body.mission_date[0];
        object_classes = req.body.object_classes[0].split(",");
        is_public = JSON.parse(req.body.is_public[0]);
        first = false;
        last = false;
        queued_filenames = req.body.queued_filenames[0].split(",");
        camera_height = req.body.camera_height[0];
        let num_sent;
        for (let i = 0; i < req.body.num_sent.length; i++) {
            num_sent = parseInt(req.body.num_sent[i]);
            if (num_sent == 1) {
                first = true;
            }
            if (num_sent == queued_filenames.length) {
                last = true;
            }
        }
        
    }
    else {
        upload_uuid = req.body.upload_uuid;
        farm_name = req.body.farm_name;
        field_name = req.body.field_name;
        mission_date = req.body.mission_date;
        object_classes = req.body.object_classes.split(",");
        is_public = JSON.parse(req.body.is_public);
        queued_filenames = req.body.queued_filenames.split(",");
        first = parseInt(req.body.num_sent) == 1;
        last = parseInt(req.body.num_sent) == queued_filenames.length;
        camera_height = req.body.camera_height;
    }

    console.log("first?", first);
    console.log("last?", last);

    
    if (first) {
        if (upload_uuid in active_uploads) {
            return res.status(422).json({
                error: "Upload key conflict."
            });
        }
        else {
            active_uploads[upload_uuid] = queued_filenames.length;
        }
    }
    else {
        if (!(upload_uuid in active_uploads)) {
            return res.status(422).json({
                error: "Upload is no longer active."
            });
        }
        else {
            if (req.files.length > 1) {
                for (let i = 0; i < req.body.queued_filenames.length; i++) {
                    let queued_filenames = req.body.queued_filenames[i].split(",");
                    if (queued_filenames.length != active_uploads[upload_uuid]) {

                        try {
                            remove_image_set(username, farm_name, field_name, mission_date);
                        }
                        catch (error) {
                            console.log("Failed to remove image set");
                            console.log(error);
                        }

                        return res.status(422).json({
                            error: "Size of image set changed during upload."
                        });
                    }
                }
            }
            else {
                let queued_filenames = req.body.queued_filenames.split(",");
                if (queued_filenames.length != active_uploads[upload_uuid]) {

                    try {
                        remove_image_set(username, farm_name, field_name, mission_date);
                    }
                    catch (error) {
                        console.log("Failed to remove image set");
                        console.log(error);
                    }

                    return res.status(422).json({
                        error: "Size of image set changed during upload."
                    });
                }
            }
        }
    }

    let image_sets_root = path.join(USR_DATA_ROOT, username, "image_sets");
    let farm_dir = path.join(image_sets_root, farm_name);
    let field_dir = path.join(farm_dir, field_name);
    let mission_dir = path.join(field_dir, mission_date);
    let images_dir = path.join(mission_dir, "images");

    
    if (first) {

        for (let filename of queued_filenames) {
            if (FILE_FORMAT.test(filename)) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "One or more provided filenames contains illegal characters."
                });
            }
            let split_filename = filename.split(".");
            if ((split_filename.length != 1) && (split_filename.length != 2)) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "At least one filename contains an illegal '.' character."
                });

            }

            let extensionless_fname = split_filename[0];
            if (extensionless_fname.length > MAX_EXTENSIONLESS_FILENAME_LENGTH) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "One or more filenames exceeds maximum allowed length of " + MAX_EXTENSIONLESS_FILENAME_LENGTH + " characters."
                });
            }
    
        }


        if (fpath_exists(mission_dir)) {
            
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "The provided farm-field-mission combination already exists."
            });
        }
        console.log("checking components");
        let id_components = [farm_name, field_name, mission_date];
        for (let id_component of id_components) {
            if (id_component.length < 3) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "The provided farm name, field name, or mission date is too short."
                });
            }
            if (id_component.length > 20) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "The provided farm name, field name, or mission date is too long."
                });
            }
            if (FARM_FIELD_MISSION_FORMAT.test(id_component)) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "The provided farm, field, or mission date contains illegal characters."
                });
            }
        }
        let date = new Date(mission_date);
        if (!(date.isValid())) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "The provided mission date is invalid."
            });
        }


        let objects_path = path.join(USR_SHARED_ROOT, "objects.json");
        let objects;
        try {
            objects = JSON.parse(fs.readFileSync(objects_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.status(422).json({
                error: "Failed to read objects file."
            });
        }
        for (let object_class of object_classes) {
            if (!(objects["object_names"].includes(object_class))) {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "One of the provided object classes ('" + object_class + "') is not recognized by the system."
                });
            }
        }


        if (camera_height.length > 0) {
            if (isNumeric(camera_height)) {
                let numeric_camera_height = parseFloat(camera_height);
                if (numeric_camera_height < MIN_CAMERA_HEIGHT || numeric_camera_height > MAX_CAMERA_HEIGHT) {
                    delete active_uploads[upload_uuid];
                    return res.status(422).json({
                        error: "The provided camera height falls outside of the accepted range."
                    });
                }
            }
            else {
                delete active_uploads[upload_uuid];
                return res.status(422).json({
                    error: "The provided camera height is invalid."
                });
            }
        }

        console.log("Making the images directory");
        fs.mkdirSync(images_dir, { recursive: true });

    }
    else {
        if (!(fpath_exists(mission_dir))) {
            try {
                remove_image_set(username, farm_name, field_name, mission_date);
            }
            catch (error) {
                console.log("Failed to remove image set");
                console.log(error);
            }
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "Image set directories were not created by initial request."
            });
        }
    }

    console.log("Writing the image files");
    for (let file_index = 0; file_index < req.files.length; file_index++) {
        let file = req.files[file_index];
        console.log(file);
        console.log(file.buffer);

        let split_filename = file.originalname.split(".");
        let extensionless_fname = split_filename[0];
        if (extensionless_fname.length > MAX_EXTENSIONLESS_FILENAME_LENGTH) {
            try {
                remove_image_set(username, farm_name, field_name, mission_date);
            }
            catch (error) {
                console.log("Failed to remove image set");
                console.log(error);
            }
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "One or more filenames exceeds maximum allowed length of " + MAX_EXTENSIONLESS_FILENAME_LENGTH + " characters."
            });
        }

        let fpath = path.join(images_dir, file.originalname);
        try {
            fs.writeFileSync(fpath, file.buffer);
        }
        catch (error) {
            console.log(error);
            try {
                remove_image_set(username, farm_name, field_name, mission_date);
            }
            catch (error) {
                console.log("Failed to remove image set");
                console.log(error);
            }
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "Error occurred when writing image file."
            });
        }
    }

    if (last) {

        let config = {
            "username": username,
            "farm_name": farm_name,
            "field_name": field_name,
            "mission_date": mission_date,
            "object_classes": object_classes,
            "is_public": is_public,
            "is_ortho": false
            
        }
        if (camera_height.length > 0) {
            config["camera_height"] = parseFloat(camera_height);
        }
        else {
            config["camera_height"] = "";
        }

        let config_path = path.join(mission_dir, "config.json");
        try {
            fs.writeFileSync(config_path, JSON.stringify(config));
        }
        catch (error) {
            delete active_uploads[upload_uuid];
            return res.status(422).json({
                error: "Error occurred when writing configuration file."
            });
        }


        let process_upload_command = "python ../../backend/src/process_upload.py " + mission_dir;
        exec(process_upload_command, {shell: "/bin/bash"}, function (error, stdout, stderr) {
            if (error) {
                console.log(error.stack);
                console.log('Error code: '+error.code);
                console.log('Signal received: '+error.signal);
            }
        });

        delete active_uploads[upload_uuid];

    }

    return res.sendStatus(200);

}


exports.post_home = async function(req, res, next) {

    let action = req.body.action;
    let username = req.session.user.username;
    let response = {};

    if (action === "get_annotations") {
        let anno_username = req.body.username; 
        let anno_farm_name = req.body.farm_name;
        let anno_field_name = req.body.field_name;
        let anno_mission_date = req.body.mission_date;
        let image_set_dir = path.join(USR_DATA_ROOT, anno_username, "image_sets", 
                                        anno_farm_name, anno_field_name, anno_mission_date);
        let annotations_path = path.join(image_set_dir, "annotations", "annotations.json");
        let annotations;
        try {
            annotations = JSON.parse(fs.readFileSync(annotations_path, 'utf8'));
        }
        catch (error) {
            response.message = "Failed to read annotations file";
            response.error = true;
            return res.json(response);
        }
        let metadata_path = path.join(image_set_dir, "metadata", "metadata.json");
        let metadata;
        try {
            metadata = JSON.parse(fs.readFileSync(metadata_path, 'utf8'));
        }
        catch (error) {
            response.message = "Failed to read metadata file";
            response.error = true;
            return res.json(response);
        }

        response.object_classes = metadata["object_classes"].join(",");
        response.annotations = annotations;
        response.error = false;
        return res.json(response);

    }
    else if (action === "destroy_model") {
        let model_name = req.body.model_name;
        let model_state = req.body.model_state;
        let models_dir = path.join(USR_DATA_ROOT, username, "models");

        if (model_state !== "available" && model_state !== "aborted") {
            response.message = "Invalid model state provided.";
            response.error = true;
            return res.json(response);
        }

        if (MODEL_NAME_FORMAT.test(model_name)) {
            response.message = "Model name contains illegal characters.";
            response.error = true;
            return res.json(response);
        }

        if ((model_name.length < 3) || (model_name.length > 50)) {
            response.message = "Illegal model name length.";
            response.error = true;
            return res.json(response);
        }

        let model_path;
        if (model_state === "available") {
            let model_public_path = path.join(models_dir, "available", "public", model_name);
            let model_private_path = path.join(models_dir, "available", "private", model_name);
            let public_path_exists;
            try {
                public_path_exists = fs.existsSync(model_public_path);
            }
            catch (error) {
                response.message = "Failed to find model.";
                response.error = true;
                return res.json(response);
            }
            if (public_path_exists) {
                model_path = model_public_path;
            }
            else {
                model_path = model_private_path;
            }
        }
        else {
            model_path = path.join(models_dir, "aborted", model_name);
        }

        try {
            fs.rmSync(model_path, { recursive: true, force: false });
        }
        catch (error) {
            response.message = "Failed to destroy model.";
            response.error = true;
            return res.json(response);
        }

        response.error = false;
        return res.json(response);

    }
    else if (action === "fetch_my_models") {

        let model_state = req.body.model_state;
        let models_dir = path.join(USR_DATA_ROOT, username, "models");

        if (model_state === "available") {

            let available_dir = path.join(models_dir, "available");

            glob(path.join(available_dir, "public", "*"), function(error, public_paths) {

                if (error) {
                    response.message = "Failed to retrieve models.";
                    response.error = true;
                    return res.json(response);
                }

                glob(path.join(available_dir, "private", "*"), function(error, private_paths) {
                    if (error) {
                        response.message = "Failed to retrieve models.";
                        response.error = true;
                        return res.json(response);
                    }

                    let models = [];

                    for (let public_path of public_paths) {

                        let log_path = path.join(public_path, "log.json");
                        let log;
                        try {
                            log = JSON.parse(fs.readFileSync(log_path, 'utf8'));
                        }
                        catch (error) {
                            response.message = "Failed to read model log.";
                            response.error = true;
                            return res.json(response);
                        }
                        models.push({
                            "log": log,
                        });
                    }
                    for (let private_path of private_paths) {

                        let log_path = path.join(private_path, "log.json");
                        let log;
                        try {
                            log = JSON.parse(fs.readFileSync(log_path, 'utf8'));
                        }
                        catch (error) {
                            response.message = "Failed to read model log.";
                            response.error = true;
                            return res.json(response);
                        }
                        models.push({
                            "log": log,
                        });
                    }
                    response.models = models;
                    response.error = false;
                    return res.json(response);
                });
            });
        }

        else if (model_state === "pending") {

            let pending_dir = path.join(models_dir, "pending");

            glob(path.join(pending_dir, "*"), function(error, model_paths) {
                if (error) {
                    response.error = true;
                    return res.json(response);
                }

                let models = [];
                for (let model_path of model_paths) {

                    let log_path = path.join(model_path, "log.json");
                    let log;
                    try {
                        log = JSON.parse(fs.readFileSync(log_path, 'utf8'));
                    }
                    catch (error) {
                        response.error = true;
                        return res.json(response);
                    }

                    models.push({
                       "log": log
                    });
                }

                response.models = models;
                response.error = false;
                return res.json(response);

            });
        }

        else if (model_state === "aborted") {
            let aborted_dir = path.join(models_dir, "aborted");

            glob(path.join(aborted_dir, "*"), function(error, model_paths) {
                if (error) {
                    response.message = "Failed to obtain list of aborted models.";
                    response.error = true;
                    return res.json(response);
                }

                let models = [];
                for (let model_path of model_paths) {
                    let log_path = path.join(model_path, "log.json");
                    let log;
                    try {
                        log = JSON.parse(fs.readFileSync(log_path, 'utf8'));
                    }
                    catch (error) {
                        response.message = "Failed to read log file of aborted model.";
                        response.error = true;
                        return res.json(response);
                    }

                    models.push({
                        "log": log
                    });
                }

                response.models = models;
                response.error = false;
                return res.json(response);

            });


        }
    }
    else if (action === "get_overview_info") {

        let farm_name = req.body.farm_name;
        let field_name = req.body.field_name;
        let mission_date = req.body.mission_date;

        let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets", 
                                    farm_name, field_name, mission_date);

        let annotations_path = path.join(image_set_dir, "annotations", "annotations.json");
        let annotations;
        try {
            annotations = JSON.parse(fs.readFileSync(annotations_path, 'utf8'));
        }
        catch (error) {
            response.message = "Failed to read annotations file";
            response.error = true;
            return res.json(response);
        }

        let annotation_info = {
            "num_annotations": 0,
            "num_images": 0,
            "num_regions_of_interest": 0,
            "num_fine_tuning_regions": 0,
            "num_test_regions": 0
        };
        for (let image_name of Object.keys(annotations)) {
            annotation_info["num_annotations"] += annotations[image_name]["boxes"].length;
            annotation_info["num_regions_of_interest"] += annotations[image_name]["regions_of_interest"].length;
            annotation_info["num_fine_tuning_regions"] += annotations[image_name]["fine_tuning_regions"].length;
            annotation_info["num_test_regions"] += annotations[image_name]["test_regions"].length;
            annotation_info["num_images"]++;
        }

        let metadata_path = path.join(image_set_dir, "metadata", "metadata.json");
        let metadata;
        try {
            metadata = JSON.parse(fs.readFileSync(metadata_path, 'utf8'));
        }
        catch (error) {
            response.message = "Failed to read metadata file";
            response.error = true;
            return res.json(response);
        }

        response.annotation_info = annotation_info;
        response.metadata = metadata;
        response.error = false;
        return res.json(response);
    }

    else if (action === "delete_image_set") {
        
        let farm_name = req.body.farm_name;
        let field_name = req.body.field_name;
        let mission_date = req.body.mission_date;

        if ((farm_name === "" || field_name === "") || mission_date === "") {
            response.message = "Could not delete the image set: an illegal farm/field/mission combination was provided.";
            response.error = true;
            return res.json(response);
        }
        if ((farm_name == null || field_name == null) || mission_date == null) {
            response.message = "Could not delete the image set: an illegal farm/field/mission combination was provided.";
            response.error = true;
            return res.json(response);
        }


        let mission_dir = path.join(USR_DATA_ROOT, username, "image_sets", 
                                    farm_name, field_name, mission_date);

        let image_set_key = ([username, farm_name, field_name, mission_date]).join("/");
        for (let socket_id of Object.keys(socket_api.workspace_id_to_key)) {
            if (socket_api.workspace_id_to_key[socket_id] === image_set_key) {
                response.message = "The image set cannot be deleted since the corresponding workspace is currently occupied. Please try again later.";
                response.error = true;
                return res.json(response);
            }
        }
            
        let annotations_path = path.join(mission_dir, "annotations", "annotations.json");
        let annotations;
        try {
            annotations = JSON.parse(fs.readFileSync(annotations_path, 'utf8'));
        }
        catch (error) {
            console.log("Annotations file does not exist. Removing image set.");
            try {
                fs.rmSync(mission_dir, { recursive: true, force: false });

                let field_dir = path.join(USR_DATA_ROOT, username, "image_sets", farm_name, field_name);
                let missions = get_subdirnames(field_dir);
                if (missions.length == 0) {
                    fs.rmSync(field_dir, { recursive: true, force: false });
                    let farm_dir = path.join(USR_DATA_ROOT, username, "image_sets", farm_name);
                    let fields = get_subdirnames(farm_dir);
                    if (fields.length == 0) {
                        fs.rmSync(farm_dir, { recursive: true, force: false });
                    }
                }
            }
            catch (error) {
                console.log(error);
                response.error = true;
                response.message = "An error occurred while deleting the image set: " + error.toString();
                return res.json(response);
            }
            response.error = false;
            response.redirect = process.env.AC_PATH + "/home/" + username;
            return res.json(response);
        }


        let empty = true;
        for (let image_name of Object.keys(annotations)) {
            if (annotations[image_name]["boxes"].length > 0) {
                empty = false;
                break;
            }
        }
        if (!(empty)) {
            response.error = true;
            response.message = "Cannot delete an image set with annotations."
            return res.json(response);
        }
        else {
            console.log("No annotations found, deleting image set");

            try {
                fs.rmSync(mission_dir, { recursive: true, force: false });

                let field_dir = path.join(USR_DATA_ROOT, username, "image_sets", farm_name, field_name);
                let missions = get_subdirnames(field_dir);
                if (missions.length == 0) {
                    fs.rmSync(field_dir, { recursive: true, force: false });
                    let farm_dir = path.join(USR_DATA_ROOT, username, "image_sets", farm_name);
                    let fields = get_subdirnames(farm_dir);
                    if (fields.length == 0) {
                        fs.rmSync(farm_dir, { recursive: true, force: false });
                    }
                }
            }
            catch (error) {
                console.log(error);
                response.error = true;
                response.message = "An error occurred while deleting the image set: " + error.toString();
                return res.json(response);
            }

            response.error = false;
            response.redirect = process.env.AC_PATH + "/home/" + username;
            return res.json(response);
        }

    }
    else if (action === "access_workspace") {

        let farm_name = req.body.farm_name;
        let field_name = req.body.field_name;
        let mission_date = req.body.mission_date;

        let image_set_key = ([username, farm_name, field_name, mission_date]).join("/");
        for (let socket_id of Object.keys(socket_api.workspace_id_to_key)) {
            if (socket_api.workspace_id_to_key[socket_id] === image_set_key) {
                console.log("The workspace is in use", image_set_key);
                response.error = true;
                response.message = "This workspace is currently in use. Please try again later.";
                return res.json(response);
            }
        }

        response.error = false;
        response.redirect = process.env.AC_PATH + "/workspace/" + username + "/" + farm_name + "/" +
                            field_name + "/" + mission_date;
        return res.json(response);
    }
    else if (action === "fetch_upload_status") {
       
        let farm_name = req.body.farm_name;
        let field_name = req.body.field_name;
        let mission_date = req.body.mission_date;


        let upload_status_path = path.join(USR_DATA_ROOT, username, "image_sets",
                                  farm_name, field_name, mission_date, "upload_status.json");

        try {
            upload_status = JSON.parse(fs.readFileSync(upload_status_path, 'utf8'));
        }
        catch (error) {
            response.error = true;
            return res.json(response);
        }
        response.error = false;
        response.status = upload_status;
        return res.json(response);

    }
    else if (action === "fetch_results") {

        let farm_name = req.body.farm_name;
        let field_name = req.body.field_name;
        let mission_date = req.body.mission_date;


        let model_dir = path.join(USR_DATA_ROOT, username, "image_sets",
                                  farm_name, field_name, mission_date, "model");
        let results_dir = path.join(model_dir, "results");

        response.aborted_results = [];
        response.completed_results = [];

        glob(path.join(results_dir, "aborted", "*"), function(error, aborted_dirs) {

            if (error) {
                console.log(error);
                response.error = true;
                return res.json(response);
            }

            for (let aborted_dir of aborted_dirs) {
                let request_path = path.join(aborted_dir, "request.json");
                try {
                    response.aborted_results.push(JSON.parse(fs.readFileSync(request_path, 'utf8')));
                }
                catch (error) {
                    console.log(error);
                    response.error = true;
                    return res.json(response);
                }
            }

            glob(path.join(results_dir, "available", "*"), function(error, completed_dirs) {
                if (error) {
                    console.log(error);
                    response.error = true;
                    return res.json(response);
                }

                for (let completed_dir of completed_dirs) {
                    let request_path = path.join(completed_dir, "request.json");
                    try {
                        response.completed_results.push(JSON.parse(fs.readFileSync(request_path, 'utf8')));
                    }
                    catch (error) {
                        console.log(error);
                        response.error = true;
                        return res.json(response);
                    }
                }

                response.error = false;
                return res.json(response);
            });
        });


    }
    else if (action === "delete_result") {
       
        let farm_name = req.body.farm_name;
        let field_name = req.body.field_name;
        let mission_date = req.body.mission_date;
        let result_type = req.body.result_type;
        let result_id = req.body.result_id;

        console.log("delete_result");
        console.log("result_type", result_type);
        console.log("result_id", result_id);

        if (result_id === "" || result_id == null) {
            response.message = "Cannot destroy result: invalid result identifier provided.";
            response.error = true;
            return res.json(response);
        }

        let result_dir;
        if (result_type === "completed") {
            result_dir = path.join(USR_DATA_ROOT, username, "image_sets",
            farm_name, field_name, mission_date, "model", "results", "available", result_id);
        }
        else {
            result_dir = path.join(USR_DATA_ROOT, username, "image_sets",
            farm_name, field_name, mission_date, "model", "results", "aborted", result_id);
        }

        try {
            fs.rmSync(result_dir, { recursive: true, force: false });
        }
        catch (error) {
            console.log(error);
            response.message = "Failed to destroy result.";
            response.error = true;
            return res.json(response);
        }

        socket_api.results_notification(username, farm_name, field_name, mission_date);

        response.error = false;
        return res.json(response);

    }
    else if (action === "add_camera") {

        let make = req.body.make;
        let model = req.body.model;
        let sensor_width = req.body.sensor_width;
        let sensor_height = req.body.sensor_height;
        let focal_length = req.body.focal_length;
        let image_width_px = req.body.image_width_px;
        let image_height_px = req.body.image_height_px;
        let farm_name = req.body.farm_name;
        let field_name = req.body.field_name;
        let mission_date = req.body.mission_date;


        let format = /[`!@#$%^&*()+\=\[\]{};':"\\|,<>\/?~]/;
        for (let input of [make, model, sensor_width, sensor_height, focal_length, image_width_px, image_height_px]) {
            if (format.test(input)) {
                response.message = "Provided metadata contains invalid characters."
                response.error = true;
                return res.json(response);
            }
        }
        for (let input of [make, model]) {
            if ((input.length < 3 || input.length > 20)) {
                response.message = "Provided metadata is invalid."
                response.error = true;
                return res.json(response);
            }
        }

        for (let input of [sensor_width, sensor_height, focal_length, image_width_px, image_height_px]) {
            if (input.length < 1 || input.length > 10) {
                response.message = "Provided metadata is invalid."
                response.error = true;
                return res.json(response);
            }
            if (!(isNumeric(input))) {
                response.message = "Provided metadata is invalid."
                response.error = true;
                return res.json(response);
            }
            input = parseFloat(input);
            if (input <= 0) {
                response.message = "Provided metadata is invalid."
                response.error = true;
                return res.json(response);
            }
        }

        for (let input of [image_width_px, image_height_px]) {
            input = parseFloat(input);
            if (!(Number.isInteger(input))) {
                return false;
            }
        }

        sensor_width = parseFloat(sensor_width);
        sensor_height = parseFloat(sensor_height);
        focal_length = parseFloat(focal_length);
        image_width_px = parseInt(image_width_px);
        image_height_px = parseInt(image_height_px);


        camera_mutex.acquire()
        .then(function(release) {

            let camera_specs;
            let camera_specs_path = path.join(USR_DATA_ROOT, username, "cameras", "cameras.json");
            try {
                camera_specs = JSON.parse(fs.readFileSync(camera_specs_path, 'utf8'));
            }
            catch (error) {
                release();
                console.log(error);
                response.message = "Failed to read camera metadata file.";
                response.error = true;
                return res.json(response);
            }

            if (!(make in camera_specs)) {
                camera_specs[make] = {};
            }

            camera_specs[make][model] = {
                "sensor_width": sensor_width,
                "sensor_height": sensor_height,
                "focal_length": focal_length,
                "image_width_px": image_width_px,
                "image_height_px": image_height_px
            }

            let metadata;
            let metadata_path = path.join(USR_DATA_ROOT, username, "image_sets",
                                         farm_name, field_name, mission_date, "metadata", "metadata.json");
            try {
                metadata = JSON.parse(fs.readFileSync(metadata_path, 'utf8'));
            }
            catch (error) {
                release();
                console.log(error);
                response.message = "Failed to read image set metadata file.";
                response.error = true;
                return res.json(response);
            }

            metadata["camera_info"]["make"] = make;
            metadata["camera_info"]["model"] = model;

            console.log("updating metadata for image_set", metadata);

            try {
                fs.writeFileSync(metadata_path, JSON.stringify(metadata));
            }
            catch (error) {
                release();
                response.message = "Failed to write image set metadata file.";
                response.error = true;
                return res.json(response);
            }

            try {
                fs.writeFileSync(camera_specs_path, JSON.stringify(camera_specs));
            }
            catch (error) {
                release();
                response.message = "Failed to write camera metadata file.";
                response.error = true;
                return res.json(response);
            }

            release();
            response.error = false;
            response.camera_specs = camera_specs;

            return res.json(response);


        }).catch(function(error) {
            console.log(error);
            response.message = "Failed to acquire camera metadata mutex.";
            response.error = true;
            return res.json(response);
        });

    }
    else if (action === "update_camera_height") {

        let farm_name = req.body.farm_name;
        let field_name = req.body.field_name;
        let mission_date = req.body.mission_date;
        let camera_height = req.body.camera_height;

        let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets",
                                      farm_name, field_name, mission_date);

        let metadata_path = path.join(image_set_dir, "metadata", "metadata.json");

        console.log("metadata_path", metadata_path);
        console.log("camera_height", camera_height);

        let camera_height_val;
        if (camera_height.length == 0) {
            camera_height_val = "";
        }
        else {
            if (!(isNumeric(camera_height))) {
                response.error = true;
                response.message = "Provided camera height is invalid.";
                return res.json(response);
            }
            camera_height_val = parseFloat(camera_height);
            if (camera_height_val < MIN_CAMERA_HEIGHT || camera_height_val > MAX_CAMERA_HEIGHT) {
                response.error = true;
                response.message = "Provided camera height is invalid.";
                return res.json(response);
            }
        }

        if (fpath_exists(metadata_path)) {
            try {
                metadata = JSON.parse(fs.readFileSync(metadata_path, 'utf8'));
            }
            catch (error) {
                response.error = true;
                response.message = "Failed to read camera metadata file.";
                return res.json(response);
            }
            metadata["camera_height"] = camera_height_val;
            try {
                fs.writeFileSync(metadata_path, JSON.stringify(metadata));
            }
            catch (error) {
                response.error = true;
                response.message = "Failed to write camera metadata file.";
                return res.json(response);
            }
            response.error = false;
            return res.json(response);

        }
        else {
            response.error = true;
            response.message = "Metadata file does not exist.";
            return res.json(response);
        }
    }
    else if (action === "train") {

        let model_name = req.body.model_name;
        if (MODEL_NAME_FORMAT.test(model_name)) {
            response.message = "Model name contains illegal characters.";
            response.error = true;
            return res.json(response);
        }

        if ((model_name.length < 3) || (model_name.length > 50)) {
            response.message = "Illegal model name length.";
            response.error = true;
            return res.json(response);
        }

        if (model_name === "Random Weights") {
            response.message = "Illegal model name.";
            response.error = true;
            return res.json(response);
        }

        let image_sets = JSON.parse(req.body.image_sets);
        if (image_sets.length == 0) {
            response.message = "At least one training image set must be supplied.";
            response.error = true;
            return res.json(response);
        }

        let model_classes = [];
        for (let image_set of Object.values(image_sets)) {
            for (let added_class_ind of image_set["added_class_indices"]) {
                let model_class = image_set["object_classes"][added_class_ind];
                if (!(model_classes.includes(model_class))) {
                    model_classes.push(model_class);
                }
            }
        }

        if (model_classes.length == 0) {
            response.message = "Training sets must contain at least one class of object.";
            response.error = true;
            return res.json(response);
        }
        if (model_classes.length > 9) {
            response.message = "Training sets cannot contain more than nine distinct object classes.";
            response.error = true;
            return res.json(response);
        }


        model_classes = nat_orderBy.orderBy(model_classes);


        let submission_image_sets = [];
        for (let image_set of Object.values(image_sets)) {
            submission_image_set = {
                "username": image_set["username"],
                "farm_name": image_set["farm_name"],
                "field_name": image_set["field_name"],
                "mission_date": image_set["mission_date"],
                "class_mapping": {}
            };
            for (let added_class_ind of image_set["added_class_indices"]) {
                let model_class = image_set["object_classes"][added_class_ind];
                let model_class_ind = model_classes.indexOf(model_class);
                submission_image_set["class_mapping"][added_class_ind] = model_class_ind;
            }
            submission_image_sets.push(submission_image_set);
        }

        /* TODO: additional error checking for image_sets */

        let models_dir = path.join(USR_DATA_ROOT, username, "models");
        let available_dir = path.join(models_dir, "available");
        let pending_dir = path.join(models_dir, "pending");
        let aborted_dir = path.join(models_dir, "aborted");

        let public_model_path = path.join(available_dir, "public", model_name);
        let private_model_path = path.join(available_dir, "private", model_name);
        let pending_model_path = path.join(pending_dir, model_name);
        let aborted_model_path = path.join(aborted_dir, model_name);

        let possible_paths = [public_model_path, private_model_path, pending_model_path, aborted_model_path];

        for (let possible_path of possible_paths) {
            let possible_path_exists;
            try {
                possible_path_exists = fs.existsSync(possible_path);
            }
            catch (error) {
                response.message = "An error occurred during model submission.";
                response.error = true;
                return res.json(response);
            }
            if (possible_path_exists) {
                response.message = "You have already created a model with the same name. Choose a different model name or delete the existing model.";
                response.error = true;
                return res.json(response);
            }
        }

        try {
            fs.mkdirSync(pending_model_path);
        }
        catch (error) {
            response.message = "An error occurred during model submission.";
            response.error = true;
            return res.json(response);
        }

        let log = {
            "model_name": req.body.model_name,
            "model_creator": username,
            "is_public": JSON.parse(req.body.is_public),
            "object_classes": model_classes,
            "image_sets": submission_image_sets,
            "submission_time": Math.floor(Date.now() / 1000),
            "training_regime": req.body.training_regime
        };

        if (req.body.training_regime === "fixed_num_epochs") {
            log["num_epochs"] = parseInt(req.body.num_epochs);
        }
        else {
            log["training_percent"] = parseFloat(req.body.training_percent);
            log["improvement_tolerance"] = parseFloat(req.body.improvement_tolerance);
        }


        let log_path = path.join(pending_model_path, "log.json");
        try {
            fs.writeFileSync(log_path, JSON.stringify(log));
        }
        catch (error) {
            try {
                fs.rmSync(pending_model_path, { recursive: true, force: false });
            }
            catch (error) {
                console.log("Failed to remove pending model after error.");
            }

            response.message = "An error occurred during model submission.";
            response.error = true;
            return res.json(response);
        }


        let job_key = username;

        let request = {
            "key": job_key,
            "task": "train",
            "request_time": Math.floor(Date.now() / 1000),
            "model_creator": username,
            "model_name": req.body.model_name,
            "is_public": JSON.parse(req.body.is_public),
            "training_regime": req.body.training_regime
        }

        if (req.body.training_regime  === "fixed_num_epochs") {
            request["num_epochs"] = parseInt(req.body.num_epochs);
        }
        else {
            request["training_percent"] = parseFloat(req.body.training_percent);
            request["improvement_tolerance"] = parseFloat(req.body.improvement_tolerance);
        }

        response = await notify_scheduler(request);
        return res.json(response);

    }
    else {
        response.message = "Invalid action specified.";
        response.error = true;
        return res.json(response);
    }
}


exports.get_viewer = function(req, res, next) {
    
    if ((req.session.user && req.cookies.user_sid) && (req.params.username === req.session.user.username)) {
        
        let username = req.session.user.username;
        let farm_name = req.params.farm_name;
        let field_name = req.params.field_name;
        let mission_date = req.params.mission_date;
        let result_uuid = req.params.result_uuid;

        console.log("result_uuid", result_uuid);


        let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets",
                                      farm_name, field_name, mission_date);

        let sel_results_dir = path.join(image_set_dir, "model", "results", "available", result_uuid);


        if (!(fpath_exists(sel_results_dir))) {
            return res.redirect(process.env.AC_PATH);
        }

        let maintenance_time = "";
        let maintenance_path = path.join(USR_SHARED_ROOT, "maintenance.json");
        if (fpath_exists(maintenance_path)) {
            try {
                maintenance_log = JSON.parse(fs.readFileSync(maintenance_path, 'utf8'));
            }
            catch (error) {
                console.log(error);
                return res.redirect(process.env.AC_PATH);
            }

            maintenance_time = maintenance_log["maintenance_time"];
        }

        let overlay_appearance;
        let overlay_appearance_path = path.join(USR_DATA_ROOT, username, "overlay_appearance.json");
        try {
            overlay_appearance = JSON.parse(fs.readFileSync(overlay_appearance_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }

        let hotkeys;
        let hotkeys_path = path.join(USR_DATA_ROOT, username, "hotkeys.json");
        try {
            hotkeys = JSON.parse(fs.readFileSync(hotkeys_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }

        let request_path = path.join(sel_results_dir, "request.json");
        let request;
        try {
            request = JSON.parse(fs.readFileSync(request_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }

        let annotations_path = path.join(sel_results_dir, "annotations.json");
        let annotations;
        try {
            annotations = JSON.parse(fs.readFileSync(annotations_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }
        let metadata_path = path.join(image_set_dir, "metadata", "metadata.json");
        let metadata;
        try {
            metadata = JSON.parse(fs.readFileSync(metadata_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }
        let camera_specs_path = path.join(USR_DATA_ROOT, username, "cameras", "cameras.json");
        let camera_specs;
        try {
            camera_specs = JSON.parse(fs.readFileSync(camera_specs_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }
        let metrics_path = path.join(sel_results_dir, "metrics.json");
        let metrics;
        try {
            metrics = JSON.parse(fs.readFileSync(metrics_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }

        let tags_path = path.join(sel_results_dir, "tags.json");
        let tags;        
        try {
            tags = JSON.parse(fs.readFileSync(tags_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }

        let excess_green_record_path = path.join(sel_results_dir, "excess_green_record.json");
        let excess_green_record;
        try {
            excess_green_record = JSON.parse(fs.readFileSync(excess_green_record_path, 'utf8'));
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }
        
        let dzi_images_dir = path.join(image_set_dir, "dzi_images");

        let dzi_image_paths = [];
        for (let image_name of Object.keys(annotations)) {
            let dzi_image_path = path.join(process.env.AC_PATH, dzi_images_dir, image_name + ".dzi");
            dzi_image_paths.push(dzi_image_path);

        }

        let areas_path = path.join(sel_results_dir, "areas.xlsx");
        let areas_spreadsheet_exists;
        try {
            areas_spreadsheet_exists = fs.existsSync(areas_path);
        }
        catch (error) {
            console.log(error);
            return res.redirect(process.env.AC_PATH);
        }

        let image_set_info = {
            "farm_name": farm_name,
            "field_name": field_name,
            "mission_date": mission_date,
            "result_uuid": result_uuid
        };


        let data = {};

        data["cur_page"] = "viewer";
        data["image_set_info"] = image_set_info;
        data["annotations"] = annotations;
        data["metadata"] = metadata;
        data["camera_specs"] = camera_specs;
        data["excess_green_record"] = excess_green_record;
        data["metrics"] = metrics;
        data["request"] = request;
        data["areas_spreadsheet_exists"] = areas_spreadsheet_exists;
        data["tags"] = tags;

        data["dzi_image_paths"] = nat_orderBy.orderBy(dzi_image_paths);
        data["overlay_appearance"] = overlay_appearance;
        data["hotkeys"] = hotkeys;
        data["maintenance_time"] = maintenance_time;

        res.render("viewer", {username: username, "data": data});
    }
    
    else {
        return res.redirect(process.env.AC_PATH);
    }
}

exports.post_viewer = function(req, res, next) {

    let response = {};

    let username = req.session.user.username;
    let farm_name = req.params.farm_name;
    let field_name = req.params.field_name;
    let mission_date = req.params.mission_date;
    let result_uuid = req.params.result_uuid;
    let action = req.body.action;

    let image_set_dir = path.join(USR_DATA_ROOT, username, "image_sets", farm_name, field_name, mission_date);
    let results_dir = path.join(image_set_dir, "model", "results", "available", result_uuid);

    if (action === "build_map") {

        let interpolated_value = req.body.interpolated_value;
        
        let prediction_dir = path.join(results_dir, "prediction");
        let maps_dir = path.join(results_dir, "maps");

        let rebuild_command = "python ../../backend/src/interpolate.py " + username + " " +
            farm_name + " " + field_name + " " + mission_date + " " + prediction_dir + 
            " " + maps_dir + " " + req.body.class_index + " " + interpolated_value;
        

        if (req.body.interpolation === "nearest") {
            rebuild_command = rebuild_command + " -nearest";
        }
        if (req.body.tile_size !== "") {
            rebuild_command = rebuild_command + " -tile_size " + req.body.tile_size;
        }
        if (req.body.interpolated_value !== "obj_density") {
            let vegetation_record_path = path.join(results_dir, "vegetation_record.json");
            rebuild_command = rebuild_command + " -vegetation_record_path " + vegetation_record_path;
        }
        
        console.log(rebuild_command);
        let result = exec(rebuild_command, {shell: "/bin/bash"}, function (error, stdout, stderr) {
            if (error) {
                console.log(error.stack);
                console.log('Error code: '+error.code);
                console.log('Signal received: '+error.signal);
                response.error = true;
            }
            else {
                response.error = false;
            }
            return res.json(response);
        });
    }
    else if (action === "retrieve_predictions") {


        let image_names = req.body.image_names.split(",");
        response.predictions = {};
        for (let image_name of image_names) {
            let prediction_path = path.join(
                results_dir, 
                "prediction",
                image_name + ".json"
            );

            if (fs.existsSync(prediction_path)) {

                let image_predictions;
                try {
                    image_predictions = JSON.parse(fs.readFileSync(prediction_path, 'utf8'));
                }
                catch (error) {
                    console.log(error);
                    response.error = true;
                    response.message = "Failed to retrieve predictions.";
                    return res.json(response);
                }
                response.predictions[image_name] = image_predictions;
            }
        }
        response.error = false;
        return res.json(response);

    }
}


exports.logout = function(req, res, next) {
    console.log("logging out");
    if (req.session.user && req.cookies.user_sid) {
        console.log("clearing cookies");
        res.clearCookie('user_sid');
        console.log("cookies cleared");
    }
    console.log("redirecting");
    return res.redirect(process.env.AC_PATH);
}