

let upload_uuid;
let dropzone_handlers = {};
let num_sent = 0;
let queued_filenames;

let upload_input_format = /[\s `!@#$%^&*()+\=\[\]{};':"\\|,<>\/?~]/;


const FILE_FORMAT = /[\s `!@#$%^&*()+\=\[\]{};':"\\|,<>\/?~]/;
const FARM_FIELD_MISSION_FORMAT = /[\s `!@#$%^&*()+\=\[\]{}.;':"\\|,<>\/?~]/;


function clear_form() {
    $("#farm_input").val("");
    $("#field_input").val("");
    $("#mission_input").val("");
    $("#camera_height_input").val("");
    $("#added_classes").empty();
    populate_object_classes();
    $("#object_input").prop("selectedIndex", -1);
    disable_green_buttons(["add_cls_button"]);

    for (let key of Object.keys(dropzone_handlers)) {
        dropzone_handlers[key].removeAllFiles();
    }
    disable_red_buttons(["remove_image_set_files"]);
    disable_red_buttons(["remove_orthomosaic_files"]);
    disable_green_buttons(["upload_button"]);
}


function disable_input() {

    global_disabled = true;

    disable_green_buttons(["upload_button"]);


    let inputs = ["farm_input", "field_input", "mission_input", "object_input", "camera_height_input"];

    for (let input of inputs) {
        $("#" + input).prop("disabled", true);
        $("#" + input).css("opacity", 0.5);
    }

    $("#added_classes tr").each(function() {
        let row_id = $(this).attr("id");
        let remove_button_id = row_id + "_remove_button";
        disable_red_buttons([remove_button_id]);
    });
    disable_green_buttons(["add_cls_button"]);
    $("#added_classes").css("opacity", 0.5);


    $("#upload_set_public").prop("disabled", true);

    $(".checkmark").css("opacity", 0.5);
    $(".checkmark").css("cursor", "default");
    $(".container").css("cursor", "default");

    $(".nav").css("pointer-events", "none");
    $(".nav").css("opacity", 0.5);

    let handler_name;
    if ($("#image_set_tab").is(":visible")) {
        handler_name = "image_set";
        disable_red_buttons(["remove_image_set_files"]);
    }
    else {
        handler_name = "orthomosaic";
        disable_red_buttons(["remove_orthomosaic_files"]);
    }

    $("#" + handler_name + "_dropzone").addClass("disabled_dropzone");
    $("#" + handler_name + "_dropzone").css("opacity", 0.7);


}



function enable_input() {

    enable_green_buttons(["upload_button"]);

    let inputs = ["farm_input", "field_input", "mission_input", "object_input", "camera_height_input"];

    for (let input of inputs) {
        $("#" + input).prop("disabled", false);
        $("#" + input).css("opacity", 1.0);
    }

    $("#added_classes tr").each(function() {
        let row_id = $(this).attr("id");
        let remove_button_id = row_id + "_remove_button";
        enable_red_buttons([remove_button_id]);
    });

    if ($("#object_input").val() !== null) {
        enable_green_buttons(["add_cls_button"]);
    }
    $("#added_classes").css("opacity", 1.0);

    $("#upload_set_public").prop("disabled", false);

    $(".checkmark").css("opacity", 1.0);
    $(".checkmark").css("cursor", "pointer");
    $(".container").css("cursor", "pointer");

    $(".nav").css("pointer-events", "all");
    $(".nav").css("opacity", 1.0);

    let handler_name;
    if ($("#image_set_tab").is(":visible")) {
        handler_name = "image_set";
    }
    else {
        handler_name = "orthomosaic";
    }

    $("#" + handler_name + "_dropzone").removeClass("disabled_dropzone");
    $("#" + handler_name + "_dropzone").css("opacity", 1.0);

}


function test_farm_name() {
    let input_val = $("#farm_input").val();
    let input_length = input_val.length;
    if (input_length == 0) {
        return [false, "A farm name must be provided."];
    }
    if (input_length < 3) {
        return [false, "The provided farm name is too short. At least 3 characters are required."];
    }
    if (input_length > 20) {
        return [false, "The provided farm name is too long. 20 characters is the maximum allowed length."];
    }
    if (FARM_FIELD_MISSION_FORMAT.test(input_val)) {
        return [false, "The provided farm name contains invalid characters. White space and most special characters are not allowed."];
    }
    return [true, ""];
}


function test_field_name() {
    let input_val = $("#field_input").val();
    let input_length = input_val.length;
    if (input_length == 0) {
        return [false, "A field name must be provided."];
    }
    if (input_length < 3) {
        return [false, "The provided field name is too short. At least 3 characters are required."];
    }
    if (input_length > 20) {
        return [false, "The provided field name is too long. 20 characters is the maximum allowed length."];
    }
    if (FARM_FIELD_MISSION_FORMAT.test(input_val)) {
        return [false, "The provided field name contains invalid characters. White space and most special characters are not allowed."];
    }
    return [true, ""];
}

function test_mission_date() {
    let input_val = $("#mission_input").val();
    let input_length = input_val.length;
    if (input_length == 0) {
        return [false, "A mission date must be provided."];
    }
    if (input_length < 3) {
        return [false, "The provided mission date is too short."];
    }
    if (input_length > 20) {
        return [false, "The provided mission date is too long."];
    }
    if (FARM_FIELD_MISSION_FORMAT.test(input_val)) {
        return [false, "The provided mission date contains invalid characters."];
    }
    let date = new Date(input_val);
    if (!(date.isValid())) {
        return [false, "The provided mission date is invalid."]
    }
    return [true, ""];
    
}

function test_model_object() {

    let num_classes = 0;
    $("#added_classes tr").each(function() {
        let row_id_pieces = $(this).attr("id").split("_");
        let object_class_ind = parseInt(row_id_pieces[row_id_pieces.length-1]);
        let object_class = objects["object_names"][object_class_ind];

        if (!(objects["object_names"].includes(object_class))) {
            return [false, "'" + object_class + "' is an invalid object class."];
        }
        num_classes++;
    });

    if (num_classes < 1) {
        return [false, "At least one object class must be provided."];
    }

    if (num_classes > 9) {
        return [false, "No more than nine object classes can be provided."];
    }

    return [true, ""];

}

function test_camera_height() {

    let camera_height = $("#camera_height_input").val();
    if (camera_height !== "") {
        if (!isNumeric(camera_height)) {
            return [false, "The provided camera height must be a numeric value."];
        }
        camera_height = parseFloat(camera_height);
        if (camera_height < MIN_CAMERA_HEIGHT) {
            return [false, "The provided camera height is too small. The height cannot be less than " + MIN_CAMERA_HEIGHT + " metres."];
        }
        if (camera_height > 1000) {
            return [false, "The provided camera height is too large. The height cannot exceed " + MAX_CAMERA_HEIGHT + " metres."];
        }
    }
    return [true, ""];
}


function update_submit() {
    let handler_name;
    let remove_all_id;
    if ($("#image_set_tab").is(":visible")) {
        handler_name = "image_set";
        remove_all_id = "remove_image_set_files";
    }
    else {
        handler_name = "orthomosaic";
        remove_all_id = "remove_orthomosaic_files";
    }


    if (dropzone_handlers[handler_name].files.length > 0) {
        enable_red_buttons([remove_all_id]);
        enable_green_buttons(["upload_button"]);

    }
    else {
        disable_red_buttons([remove_all_id]);
        disable_green_buttons(["upload_button"]);
    }
}




function create_orthomosaic_dropzone() {


    if (dropzone_handlers["orthomosaic"]) {
        dropzone_handlers["orthomosaic"].destroy();
    }

    $("#orthomosaic_tab").empty();

    $("#orthomosaic_tab").append(
        `<table>` +
            `<tr>` +
                `<td style="width: 100%"></td>` +
                `<td>` +
                    `<div id="remove_orthomosaic_files" class="button-red button-red-hover" style="width: 140px; font-size: 14px; padding: 2px; margin: 2px" onclick="remove_all_files()">` +
                        `<i class="fa-solid fa-circle-minus" style="padding-right: 5px"></i>` +
                            `Remove All Files` +
                    `</div>` +
                `</td>` +
            `</tr>` +
        `</table>` +
        `<div id="orthomosaic_dropzone" class="dropzone" style="height: 335px">` +
            `<div class="dz-message data-dz-message">` +
                `<span>Drop Orthomosaic Here</span>` +
            `</div>` +
            `<div id="orthomosaic_upload_loader" class="loader" hidden></div>` +
        `</div>`
    );
    disable_red_buttons(["remove_orthomosaic_files"]);

    dropzone_handlers["orthomosaic"] = new Dropzone("#orthomosaic_dropzone", { 
        url: get_AC_PATH() + "/orthomosaic_upload",
        autoProcessQueue: false,
        paramName: function(n) { return 'source_file[]'; },
        uploadMultiple: false,
        chunking: true,
        forceChunking: true,
        chunkSize: 20000000,
        parallelChunkUploads: false,
        retryChunks: false,
        retryChunksLimit: 3,
        farm_name: '',
        field_name: '',
        mission_date: '',
        maxFilesize: 200000,
        addRemoveLinks: true,
        dictRemoveFile: "Remove File",
        dictCancelUpload: ""
    });

}

function remove_all_files() {
    if (!(global_disabled)) {
        if ($("#image_set_tab").is(":visible")) {
            dropzone_handlers["image_set"].removeAllFiles(true);
        }
        else {
            dropzone_handlers["orthomosaic"].removeAllFiles(true);
        }
    }
}

function create_image_set_dropzone() {


    if (dropzone_handlers["image_set"]) {
        dropzone_handlers["image_set"].destroy();
    }

    $("#image_set_tab").empty();

    $("#image_set_tab").append(
        `<table>` +
            `<tr>` +
                `<td style="width: 100%"></td>` +
                `<td>` +
                    `<div id="remove_image_set_files" class="button-red button-red-hover" style="width: 140px; font-size: 14px; padding: 2px; margin: 2px" onclick="remove_all_files()">` +
                        `<i class="fa-solid fa-circle-minus" style="padding-right: 5px"></i>` +
                            `Remove All Files` +
                    `</div>` +
                `</td>` +
            `</tr>` +
        `</table>` +
        `<div id="image_set_dropzone" class="dropzone" style="height: 335px">` +
            `<div class="dz-message data-dz-message">` +
                `<span>Drop Images Here</span>` +
            `</div>` +
            `<div id="image_set_upload_loader" class="loader" hidden></div>` +
        `</div>`
    );
    disable_red_buttons(["remove_image_set_files"]);

    dropzone_handlers["image_set"] = new Dropzone("#image_set_dropzone", { 
        url: get_AC_PATH() + "/image_set_upload",
        autoProcessQueue: false,
        paramName: function(n) { return 'source_file[]'; },
        uploadMultiple: true,
        farm_name: '',
        field_name: '',
        mission_date: '',
        parallelUploads: 10,
        maxUploads: 10000,
        maxFilesize: 450,
        addRemoveLinks: true,
        dictRemoveFile: "Remove File",
        dictCancelUpload: "",
    });

}

function add_dropzone_listeners() {

    for (let key of Object.keys(dropzone_handlers)) {


        dropzone_handlers[key].on("success", function(file, response) {   

            dropzone_handlers[key].removeFile(file);
            if (dropzone_handlers[key].getAcceptedFiles().length == 0) {

                dropzone_handlers[key].removeAllFiles(true);
                num_sent = 0;
                dropzone_handlers[key].options.autoProcessQueue = false;

                show_modal_message(`Success!`, `<div align="center">Your image set has been successfully uploaded.<br>Additional processing is now being performed.` +
                `<br><br>The image set can now be viewed in the <i>Browse</i> tab.</div>`);

                let uploaded_farm = $("#farm_input").val();
                let uploaded_field = $("#field_input").val();
                let uploaded_mission = $("#mission_input").val();
                if (!(uploaded_farm in image_sets_data)) {
                    image_sets_data[uploaded_farm] = {};
                }
                if (!(uploaded_field in image_sets_data[uploaded_farm])) {
                    image_sets_data[uploaded_farm][uploaded_field] = {};
                }
                image_sets_data[uploaded_farm][uploaded_field][uploaded_mission] = {
                    "status": "processing"
                };

                initialize_browse();
                clear_form();
                enable_input();
                update_submit();
                global_disabled = false;

                $("#" + key + "_upload_loader").hide();
            }
        });

        dropzone_handlers[key].on("error", function(file, response) {

            let upload_error;
            if (typeof(response) == "object" && "error" in response) {
                upload_error = response.error;
            }
            else {
                upload_error = response;
            }

        
            num_sent = 0;
            dropzone_handlers[key].options.autoProcessQueue = false;
            dropzone_handlers[key].removeAllFiles(true);
        
            show_modal_message(`Error`, upload_error);
            enable_input();
            update_submit();
            global_disabled = false;
            $("#" + key + "_upload_loader").hide();

        });

        dropzone_handlers[key].on("removedfile", function(file) {
            if (!(global_disabled)) {
                update_submit();
            }
        });

        dropzone_handlers[key].on("addedfile", function() {

            if (dropzone_handlers[key].options.autoProcessQueue) {
                let upload_error = "A file was added after the upload was initiated. Please ensure that all files have been added to the queue before pressing the 'Upload' button."
                dropzone_handlers[key].removeAllFiles(true);


                if ($("#image_set_tab").is(":visible")) {
                    create_image_set_dropzone();
                }
                else {
                    create_orthomosaic_dropzone();
                }
            
                num_sent = 0;
                dropzone_handlers[key].options.autoProcessQueue = false;
            
                show_modal_message(`Error`, 
                    `<div>${upload_error}</div>` +
                    `<div style="height: 10px"></div>` +
                    `<div style="text-align: center">` +
                        `<button class="button-green button-green-hover" onclick="window.location.reload()" style="width: 150px">Reload Page</button>` +
                    `</div>`
                );
                $("#modal_close").hide();
                clear_form();
                enable_input();
                update_submit();
                global_disabled = false;
                $("#" + key + "_upload_loader").hide();
            }
            else {
                if ($("#image_set_tab").is(":visible")) {
                    enable_red_buttons(["remove_image_set_files"]);
                }
                else {
                    enable_red_buttons(["remove_orthomosaic_files"]);
                }

                $("form").change();
            }
        });


        dropzone_handlers[key].on('sending', function(file, xhr, formData) {
            let object_classes = [];
            $("#added_classes tr").each(function() {
                let row_id_pieces = $(this).attr("id").split("_");
                let object_class_ind = parseInt(row_id_pieces[row_id_pieces.length-1]);
                let object_class = objects["object_names"][object_class_ind];
                object_classes.push(object_class);
            });

            formData.append('farm_name', $("#farm_input").val());
            formData.append('field_name', $("#field_input").val());
            formData.append('mission_date', $("#mission_input").val());
            formData.append("object_classes", object_classes.join(","));
            formData.append("is_public", ($("#upload_set_public").is(':checked')) ? "yes" : "no");
            formData.append("queued_filenames",  queued_filenames.join(","));
            formData.append('camera_height', $("#camera_height_input").val());
            if (num_sent == 0) {
                upload_uuid = uuidv4();
            }
            formData.append('upload_uuid', upload_uuid);
            num_sent++;
            formData.append("num_sent", num_sent.toString());

        });
    }
}



function show_upload_tab(active_tab_btn) {

    let tab_ids = [
        "image_set_tab_btn",
        "orthomosaic_tab_btn"
    ];

    for (let tab_btn_id of tab_ids) {
        let tab_id = tab_btn_id.substring(0, tab_btn_id.length - 4);
        $("#" + tab_id).hide();
        $("#" + tab_btn_id).removeClass("tab-btn-active");
    }

    $("#" + active_tab_btn).addClass("tab-btn-active");

    $("#image_set_tab").hide();
    $("#orthomosaic_tab").hide();

    if (active_tab_btn === "image_set_tab_btn") {
        if (!global_disabled) {
            $("#image_set_tab").show();
            update_submit();
        }
    }
    else {
        if (!global_disabled) {
            $("#orthomosaic_tab").show();
            update_submit();
        }
    }

}






function populate_object_classes() {

    let added_class_inds = [];
    $("#added_classes tr").each(function() {
        let row_id_pieces = $(this).attr("id").split("_");
        added_class_inds.push(parseInt(row_id_pieces[row_id_pieces.length-1]));
    });

    $("#object_input").empty();

    if (added_class_inds.length < 9) {
        for (let i = 0; i < objects["object_names"].length; i++) {
            if (!(added_class_inds.includes(i))) {
                let object_name = objects["object_names"][i];
                $("#object_input").append($('<option>', {
                    value: object_name,
                    text: object_name
                }));
            }
        }
    }

    $("#object_input").prop("selectedIndex", -1);
    disable_green_buttons(["add_cls_button"]);

}

function remove_object_class(cls_ind) {

    let cur_obj_input = $("#object_input").val();
    
    let row_id = "upload_cls_" + cls_ind;
    $("#" + row_id).remove();

    populate_object_classes();

    $("#object_input").val(cur_obj_input);
    if (cur_obj_input !== null) {
        enable_green_buttons(["add_cls_button"]);
    }
}


function add_cls_row(cls_name) {

    let cls_ind = (objects["object_names"]).indexOf(cls_name);
    let row_id = "upload_cls_" + cls_ind;
    let remove_button_id = row_id + "_remove_button";

    $("#added_classes").append(
        `<tr style="border-bottom: 1px solid #4c6645; height: 40px" id="${row_id}">` + 
            `<td style="width: 100%">` +
            `</td>` +
            `<td>` +
                `<div style="width: 200px" class="object_entry">${cls_name}</div>` +
            `</td>` +
            `<td>` +
                `<div style="width: 75px"></div>` +
            `</td>` +
            `<td>` +
                `<button id="${remove_button_id}" onclick="remove_object_class('${cls_ind}')" class="button-red button-red-hover" style="width: 25px; height: 25px; border-radius: 5px; font-size: 12px;">` +
                    `<i class="fa-solid fa-circle-minus"></i>` + 
                `</button>` + 
            `</td>` +
            `<td><div style="width: 15px"></div></td>` +
        `</tr>` 
    );
}

function initialize_upload() {

    create_image_set_dropzone();
    create_orthomosaic_dropzone();

    add_dropzone_listeners();

    global_disabled = false;
    disable_green_buttons(["upload_button"]);

    populate_object_classes();



    $("#object_input").change(function() {
        enable_green_buttons(["add_cls_button"]);
    });

    $("#add_cls_button").click(function() {
        let cls_name = $("#object_input").val();
        add_cls_row(cls_name);
        populate_object_classes();
    });



    $("#upload_button").click(function(e) {
        e.preventDefault();
        e.stopPropagation();

        let handler_name;
        if ($("#image_set_tab").is(":visible")) {
            handler_name = "image_set";
        }
        else {
            handler_name = "orthomosaic";
        }



        disable_input();
        $("#" + handler_name + "_upload_loader").show();

        queued_filenames = [];

        if (handler_name === "orthomosaic" && dropzone_handlers[handler_name].getQueuedFiles().length != 1) {
            show_modal_message(`Error`, `Only one orthomosaic can be uploaded at a time.`);
            enable_input();
            update_submit();
            global_disabled = false;
            $("#" + handler_name + "_upload_loader").hide();
            return;
        }
        let res;
        res = test_farm_name();
        if (res[0]) {
            res = test_field_name();
        }
        if (res[0]) {
            res = test_mission_date();
        }
        if (res[0]) {
            res = test_model_object();
        }
        if (res[0]) {
            res = test_camera_height();
        }
        if (res[0]) {
            for (let f of dropzone_handlers[handler_name].getQueuedFiles()) {
                if (FILE_FORMAT.test(f.name)) {
                    res = [false, "One or more filenames contains illegal characters. White space and most special characters are not allowed."];
                }
            }
        }
        if (res[0]) {
            for (let f of dropzone_handlers[handler_name].getQueuedFiles()) {
                queued_filenames.push(f.name);
            }
            if (has_duplicates(queued_filenames)) {
                res = [false, "The image set contains duplicate filenames."];
            }
        }

        if (!(res[0])) {
            queued_filenames = [];
            show_modal_message(`Error`, res[1]);
            enable_input();
            update_submit();
            global_disabled = false;
            $("#" + handler_name + "_upload_loader").hide();
            return;
        }

        $("#" + handler_name + "_dropzone").animate({ scrollTop: 0 }, "fast");
        dropzone_handlers[handler_name].options.autoProcessQueue = true;
        dropzone_handlers[handler_name].processQueue();

    });

    $("#upload_form").change(function() {
        update_submit();
    });


    $("#image_set_tab_btn").click(function() {
        show_upload_tab("image_set_tab_btn");
    });

    $("#orthomosaic_tab_btn").click(function() {
        show_upload_tab("orthomosaic_tab_btn");
    });
}
