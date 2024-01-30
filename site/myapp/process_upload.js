const path = require('path');
const fs = require('fs');
const { spawn, exec, execSync, fork } = require('child_process');
const https = require('https');
const glob = require("glob");
const nat_orderBy = require('natural-orderby');

const USR_DATA_ROOT = path.join("usr", "data");



function write_upload_status(upload_status_path, upload_status) {
    try {
        fs.writeFileSync(upload_status_path, JSON.stringify(upload_status));
    }
    catch (error) {
        console.log(error);
    }

}

function write_and_notify(upload_status_path, upload_status, notify_data) {
    write_upload_status(upload_status_path, upload_status)
    upload_notify(notify_data["username"], notify_data["farm_name"], notify_data["field_name"], notify_data["mission_date"]);
}


function upload_notify(username, farm_name, field_name, mission_date) {

    console.log("attempting to notify the server");

    let data = JSON.stringify({
        username: username,
        farm_name: farm_name,
        field_name: field_name,
        mission_date: mission_date
    });

    let options = {
        hostname: process.env.AC_IP,
        port: parseInt(process.env.AC_PORT),
        path: process.env.AC_PATH + '/upload_notification',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': data.length,
        },
        rejectUnauthorized: false
    };

    let req = https.request(options, res => {
        console.log(`statusCode: ${res.statusCode}`);

        res.on("data", d => {
            process.stdout.write(d);
        });
    });

    req.on("error", error => {
        console.log(error);
    });

    req.write(data);
    req.end();
}



async function process_upload(username, farm_name, field_name, mission_date, object_classes_str, camera_height, is_public, is_ortho) {

    let notify_data = {
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date
    }

    console.log("processing upload");

    let image_sets_root = path.join(USR_DATA_ROOT, username, "image_sets");
    let farm_dir = path.join(image_sets_root, farm_name);
    let field_dir = path.join(farm_dir, field_name);
    let mission_dir = path.join(field_dir, mission_date);

    let upload_status_path = path.join(mission_dir, "upload_status.json");


    write_upload_status(upload_status_path, {"status": "processing"});

    let images_dir = path.join(mission_dir, "images");
    let dzi_images_dir = path.join(mission_dir, "dzi_images");
    let annotations_dir = path.join(mission_dir, "annotations");
    let metadata_dir = path.join(mission_dir, "metadata");
    
    let patches_dir = path.join(mission_dir, "patches");
    let excess_green_dir = path.join(mission_dir, "excess_green");

    let model_dir = path.join(mission_dir, "model");
    let training_dir = path.join(model_dir, "training");
    let prediction_dir = path.join(model_dir, "prediction");
    let image_requests_dir = path.join(prediction_dir, "image_requests");
    let image_set_requests_dir = path.join(prediction_dir, "image_set_requests");
    let pending_dir = path.join(image_set_requests_dir, "pending");
    let aborted_dir = path.join(image_set_requests_dir, "aborted");
    let weights_dir = path.join(model_dir, "weights");
    let results_dir = path.join(model_dir, "results");


    glob(path.join(images_dir, "*"), function(error, image_paths) {
        if (error) {
            write_and_notify(upload_status_path, {"status": "failed", "error": error.toString()}, notify_data);
            return;
        }

        let image_names = [];
        for (let image_path of image_paths) {
            let full_image_name = path.basename(image_path);
            let split_image_name = full_image_name.split(".");
            let extensionless_fname = split_image_name[0];
            image_names.push(extensionless_fname);
        }


        let check_channels_command = "python ../../backend/src/check_channels.py " + mission_dir + " " + is_ortho;
        try {
            execSync(check_channels_command, {shell: "/bin/bash"});
        }
        catch (error) {
            console.log("An error occured while running check_channels.py", error);
            let error_message;
            if (error.status == 1) {
                error_message = "At least one file is not an accepted image type. Accepted image types are JPEGs, PNGs, and TIFFs. Extensions are optional, but must match the underlying file type if they are included.";
            }
            else if (error.status == 2) {
                error_message = "At least one file's extension does not match the true underlying image type. Accepted image types are JPEGs, PNGs, and TIFFs. Extensions are optional, but must match the underlying file type if they are included.";
            }
            else if (error.status == 3) {
                error_message = "At least one image contains an invalid number of channels. Only RGB images can be uploaded (with optional alpha channel).";
            }
            else {
                error_message = "An error occurred while checking the number of channels in each image";
            }
            write_and_notify(upload_status_path, {"status": "failed", "error": error_message}, notify_data);
            return;
        }


        console.log("Creating image set directories");
        try {
            fs.mkdirSync(dzi_images_dir, { recursive: true });
            fs.mkdirSync(annotations_dir, { recursive: true });
            fs.mkdirSync(metadata_dir, { recursive: true });
            fs.mkdirSync(patches_dir, { recursive: true });
            fs.mkdirSync(excess_green_dir, { recursive: true });
            fs.mkdirSync(model_dir, { recursive: true });
            fs.mkdirSync(training_dir, { recursive: true });
            fs.mkdirSync(prediction_dir, { recursive: true });
            fs.mkdirSync(image_requests_dir, { recursive: true });
            fs.mkdirSync(image_set_requests_dir, { recursive: true });
            fs.mkdirSync(pending_dir, { recursive: true });
            fs.mkdirSync(aborted_dir, { recursive: true });
            fs.mkdirSync(weights_dir, { recursive: true });
            fs.mkdirSync(results_dir, { recursive: true});
        }
        catch (error) {
            write_and_notify(upload_status_path, {"status": "failed", "error": error.toString()}, notify_data);
            return;
        }

        let status = {
            "model_name": "---",
            "model_creator": "---",
            "num_regions_fully_trained_on": 0
        };

        let status_path = path.join(model_dir, "status.json");
        try {
            fs.writeFileSync(status_path, JSON.stringify(status));
        }
        catch (error) {
            write_and_notify(upload_status_path, {"status": "failed", "error": error.toString()}, notify_data);
            return;
        }


        console.log("Making the annotations file");
        let annotations_path = path.join(annotations_dir, "annotations.json");
        let annotations = {};
        for (let image_name of image_names) {
            annotations[image_name] = {
                "boxes": [],
                "classes": [],
                "regions_of_interest": [],
                "training_regions": [],
                "test_regions": [],
                "source": "NA"
            }
        }

        console.log("Writing the annotations file");
        try {
            fs.writeFileSync(annotations_path, JSON.stringify(annotations));
        }
        catch (error) {
            write_and_notify(upload_status_path, {"status": "failed", "error": error.toString()}, notify_data);
            return;
        }


        let tags = {};
        let tags_path = path.join(annotations_dir, "tags.json");
        try {
            fs.writeFileSync(tags_path, JSON.stringify(tags));
        }
        catch (error) {
            write_and_notify(upload_status_path, {"status": "failed", "error": error.toString()}, notify_data);
            return;
        }

        let loss_record_path = path.join(training_dir, "loss_record.json")
        let loss_record = {
            "training_loss": { "values": [],
                            "best": 100000000,
                            "epochs_since_improvement": 100000000}, 
            "validation_loss": {"values": [],
                                "best": 100000000,
                                "epochs_since_improvement": 100000000}
        }
        try {
            fs.writeFileSync(loss_record_path, JSON.stringify(loss_record));
        }
        catch (error) {
            write_and_notify(upload_status_path, {"status": "failed", "error": error.toString()}, notify_data);
            return;
        }


        console.log("Collecting metadata...");
        console.log("camera_height", camera_height);
        let metadata_command = "python ../../backend/src/metadata.py " + mission_dir;
        if (camera_height.length > 0) {
            metadata_command = metadata_command + " --camera_height " + camera_height;
        }
        console.log(metadata_command);
        try {
            execSync(metadata_command, {shell: "/bin/bash"});
        }
        catch (error) {
            console.log("Error occured during metadata extraction", error);
            let error_message;
            if (error.status == 1) {
                error_message = "The images in the image set were captured by several different camera types. This is not allowed.";
            }
            else if (error.status == 2) {
                error_message = "The images in the image set were captured by several different camera types. This is not allowed.";
            }
            else {
                error_message = "An error occurred while extracting image metadata.";
            }
            write_and_notify(upload_status_path, {"status": "failed", "error": error_message}, notify_data);
            return;
        }


        let metadata_path = path.join(mission_dir, "metadata", "metadata.json");
        let metadata;
        try {
            metadata = JSON.parse(fs.readFileSync(metadata_path, 'utf8'));
        }
        catch (error) {
            write_and_notify(upload_status_path, {"status": "failed", "error": error.toString()}, notify_data);
            return;
        }


        let excess_green_record = {};
        for (let image_name of image_names) {
            excess_green_record[image_name] = {}
            excess_green_record[image_name]["sel_val"] = 0;
        }
        let excess_green_record_path = path.join(mission_dir, "excess_green", "record.json");
        try {
            fs.writeFileSync(excess_green_record_path, JSON.stringify(excess_green_record));
        }
        catch (error) {
            write_and_notify(upload_status_path, {"status": "failed", "error": error.toString()}, notify_data);
            return;
        }

        let vegetation_record = {};
        let vegetation_record_path = path.join(mission_dir, "excess_green", "vegetation_record.json");
        try {
            fs.writeFileSync(vegetation_record_path, JSON.stringify(vegetation_record));
        }
        catch (error) {
            write_and_notify(upload_status_path, {"status": "failed", "error": error.toString()}, notify_data);
            return;
        }




        metadata["is_public"] = is_public;
        metadata["is_ortho"] = is_ortho;

        let object_classes = nat_orderBy.orderBy(object_classes_str.split("."));
        metadata["object_classes"] = object_classes;
        try {
            fs.writeFileSync(metadata_path, JSON.stringify(metadata));
        }
        catch (error) {
            write_and_notify(upload_status_path, {"status": "failed", "error": error.toString()}, notify_data);
            return;
        }


        console.log("Performing DZI image conversion");


        let slicer = spawn("python3", ["../../backend/src/slice.py", mission_dir]);



        
        slicer.stderr.on('data', (data) => {
            console.error(`Slicer stderr: ${data}`);
        });

        slicer.stdout.on('data', (data) => {
            console.log(`Slicer stdout: ${data}`);
        });

        slicer.on('error', (error) => {
            console.log("Failed to start slicer subprocess.");
            console.log(error);
        });


        slicer.on('close', (code) => {
            console.log("Slicer finished with code", code);
        
            write_and_notify(upload_status_path, {"status": "uploaded"}, notify_data);
        
            console.log("Finished processing image set");
        
            return;

        });
    });
}



let username = process.argv[2]
let farm_name = process.argv[3];
let field_name = process.argv[4];
let mission_date = process.argv[5];
let object_classes_str = process.argv[6];
let camera_height = process.argv[7];
let is_public = process.argv[8];
let is_ortho = process.argv[9];

process_upload(username, 
               farm_name, 
               field_name, 
               mission_date, 
               object_classes_str, 
               camera_height,
               is_public,
               is_ortho);