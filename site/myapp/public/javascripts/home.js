
let image_sets_data;
let camera_specs;
let objects;
let available_image_sets;
let overlay_appearance;
let viewing = {
    "browse": null,
    "train": null
};
let active_results_tab_btn = "completed_results_tab_btn";
let proposed_camera_height;
let metadata;

global_disabled = false;



function delete_request() {

    show_modal_message(`Are you sure?`, 
        `<div style="height: 30px">Are you sure you want to delete this image set?</div>` +
        `<div style="height: 20px"></div>` +
        `<div id="modal_button_container" style="text-align: center">` +
            `<button id="confirm_delete" class="button-red button-red-hover" `+
                                    `style="width: 150px" onclick="confirmed_delete_request()">Delete</button>` +
            `<div style="display: inline-block; width: 10px"></div>` +
            `<button id="cancel_delete" class="button-green button-green-hover" ` +
                                    `style="width: 150px" onclick="close_modal()">Cancel</button>` +
            `<div style="height: 20px" id="loader_container"></div>` +
        `</div>`
    );
}

function confirmed_delete_request() {


    $("#modal_close").off('click').on('click', function() {
        // do nothing
    });
    disable_red_buttons(["confirm_delete"]);
    disable_green_buttons(["cancel_delete"]);
    $("#loader_container").append(
        `<div class="loader"></div>`
    );

    
    $.post($(location).attr('href'),
    {
        action: "delete_image_set",
        farm_name: $("#farm_combo").val(),
        field_name: $("#field_combo").val(),
        mission_date: $("#mission_combo").val(),
    },
    function(response, status) {

        if (response.error) {
            show_modal_message(`Error`, response.message);
            $("#modal_close").off('click').on('click', function() {
                close_modal();
            });
        }
        else {
            window.location.href = response.redirect;
        }
    });
}


function workspace_request() {


    $.post($(location).attr('href'),
    {
        action: "access_workspace",
        farm_name: $("#farm_combo").val(),
        field_name: $("#field_combo").val(),
        mission_date: $("#mission_combo").val(),
    },
    function(response, status) {

        if (response.error) {
            show_modal_message(`Denied`, response.message);
        }
        else {
            reset_to_initial_apperance();
            window.location.href = response.redirect;
        }
    });
}


function show_results_tab() {

    let tab_ids = [
        "completed_results_tab_btn",
        "aborted_results_tab_btn"
    ];

    for (let tab_btn_id of tab_ids) {
        let tab_id = tab_btn_id.substring(0, tab_btn_id.length - 4);
        $("#" + tab_id).hide();
        $("#" + tab_btn_id).removeClass("tab-btn-active");
    }

    $("#" + active_results_tab_btn).addClass("tab-btn-active");

    $("#completed_results").hide();
    $("#aborted_results").hide();

    if (active_results_tab_btn === "completed_results_tab_btn") {
        $("#completed_results").show();
    }
    else {
        $("#aborted_results").show();
    }

}


function show_tab(sel_tab_btn_id) {

    let tab_ids = [
        "browse_tab_btn",
        "upload_tab_btn",
        "train_tab_btn"
    ];

    for (let tab_btn_id of tab_ids) {
        let tab_id = tab_btn_id.substring(0, tab_btn_id.length - 4);
        $("#" + tab_id).hide();
        $("#" + tab_btn_id).removeClass("tab-btn-active");
    }

    $("#" + sel_tab_btn_id).addClass("tab-btn-active");

    $("#browse").hide();
    $("#upload").hide();
    $("#train").hide();

    if (sel_tab_btn_id === "browse_tab_btn") {
        $("#browse").show();
    }
    else if (sel_tab_btn_id === "upload_tab_btn") {
        $("#upload").show();
    }
    else {
        $("#train").show();
    }
}

function show_image_set_tab(sel_tab_btn_id) {

    let image_set_tab_ids = [
        "overview_tab_btn",
        "results_tab_btn"
    ];

    for (let tab_btn_id of image_set_tab_ids) {
        let tab_id = tab_btn_id.substring(0, tab_btn_id.length - 4);
        $("#" + tab_id).hide();
        $("#" + tab_btn_id).removeClass("tab-btn-active");
    }

    $("#" + sel_tab_btn_id).addClass("tab-btn-active");

    if (sel_tab_btn_id === "overview_tab_btn") {
        show_overview();
    }
    else {
        fetch_and_show_results();
    }
}



function update_make_model() {

    let inputs_to_check = ["make_input", "model_input"];
    for (let input of inputs_to_check) {
        let input_val = $("#" + input).val();
        let input_length = input_val.length;
        if ((input_length < 3) || (input_length > 20)) {
            return false;
        }
    }
    return true;
}

function update_sensor() {

    let inputs_to_check = ["sensor_width_input", "sensor_height_input", "focal_length_input", "image_width_px_input", "image_height_px_input"];
    for (let input of inputs_to_check) {
        let input_length = ($("#" + input).val()).length;
        if ((input_length < 1) || (input_length > 10)) {
            return false;
        }
        let input_val = $("#" + input).val();
        if (!(isNumeric(input_val))) {
            return false;
        }
        input_val = parseFloat(input_val);
        if (input_val <= 0) {
            return false;
        }
        if ((input === "image_width_px_input") || (input === "image_height_px_input")) {
            if (!(Number.isInteger(input_val))) {
                return false;
            }
        }
    }

    return true;
}

function edit_metadata(make, model) {

    let message = `<div style="font-weight: bold">Warning:</div>` +
                  `<div style="text-align: justify; margin: 0px 20px">` +
                  `Changing a camera's metadata will affect all of the image sets that have been assigned to that camera.</div>` +
                  `<div style="height: 20px"></div>`;

    message = message + add_make_model_fields("200px");
    message = message + add_sensor_fields("200px");
    

    let sensor_width = camera_specs[make][model]["sensor_width"];
    let sensor_height = camera_specs[make][model]["sensor_height"];
    let focal_length = camera_specs[make][model]["focal_length"];
    let image_width_px = camera_specs[make][model]["image_width_px"];
    let image_height_px = camera_specs[make][model]["image_height_px"];

 
    message = message + `<div style="height: 20px"></div>` +
    `<div style="text-align: center">` +
        `<button class="button-black button-black-hover" style="width: 220px; height: 30px;" id="camera_update_button" ` +
        `onclick="update_camera()">`+
            `Update Metadata</button>` +
    `</div>`;

    show_modal_message(`Edit Camera Metadata`, message);
    $("#make_input").val(make);
    $("#model_input").val(model);
    $("#sensor_width_input").val(sensor_width);
    $("#sensor_height_input").val(sensor_height);
    $("#focal_length_input").val(focal_length);
    $("#image_width_px_input").val(image_width_px);
    $("#image_height_px_input").val(image_height_px);

    for (let input_id of ["make_input", "model_input", "sensor_width_input", "sensor_height_input", "focal_length_input", "image_width_px_input", "image_height_px_input"]) {
        $("#" + input_id).on("input", function(e) {
            if (update_make_model() && update_sensor()) {
                enable_buttons(["camera_update_button"]);
            }
            else {
                disable_buttons(["camera_update_button"]);
            }
        });
    }
}

function add_sensor_fields(left_col_width_px) {

    let message =
        `<table class="transparent_table">` +
        `<tr>` +
            `<td>` + 
                `<div class="table_head" style="width: ${left_col_width_px}; padding-right: 10px">Sensor Width (mm)</div>` +
            `</td>` +
            `<td>` +
                `<div style="width: 250px">` +
                    `<input id="sensor_width_input" class="nonfixed_input">` +
                `</div>` +
            `</td>` +
        `</tr>` + 
        `<tr>` +
            `<td>` + 
                `<div class="table_head" style="width: ${left_col_width_px}; padding-right: 10px">Sensor Height (mm)</div>` +
            `</td>` +
            `<td>` +
                `<div style="width: 250px">` +
                    `<input id="sensor_height_input" class="nonfixed_input">` +
                `</div>` +
            `</td>` +
        `</tr>` +
        `<tr>` +
            `<td>` + 
                `<div class="table_head" style="width: ${left_col_width_px}; padding-right: 10px">Focal Length (mm)</div>` +
            `</td>` +
            `<td>` +
                `<div style="width: 250px">` +
                    `<input id="focal_length_input" class="nonfixed_input">` +
                `</div>` +
            `</td>` +
        `</tr>` +
        `<tr>` +
            `<td>` + 
                `<div class="table_head" style="width: ${left_col_width_px}; padding-right: 10px">Raw Image Width (pixels)</div>` +
            `</td>` +
            `<td>` +
                `<div style="width: 250px">` +
                    `<input id="image_width_px_input" class="nonfixed_input">` +
                `</div>` +
            `</td>` +
        `</tr>` +
        `<tr>` +
            `<td>` + 
                `<div class="table_head" style="width: ${left_col_width_px}; padding-right: 10px">Raw Image Height (pixels)</div>` +
            `</td>` +
            `<td>` +
                `<div style="width: 250px">` +
                    `<input id="image_height_px_input" class="nonfixed_input">` +
                `</div>` +
            `</td>` +
        `</tr>` +


        `</table>`;

    return message;
}

function add_sensor_metadata(make, model) {

    let message = `<div>` +
    `This camera is not known to the system.<br>` +
    `Please provide the following information:</div>`;
    message = message + `<div style="height: 20px"></div>`;


    message = message + add_sensor_fields("200px");

    message = message + `<div style="height: 20px"></div>`;

    message = message + `<table class="transparent_table"><tr>` +
    `<td>` +
    `<button class="button-black button-black-hover" style="width: 220px; height: 30px;" id="camera_add_button" ` +
    `onclick="add_camera('${make}', '${model}')">`+
        `Add Camera</button>` +
    `</td>` +
    `</tr></table>`;

    show_modal_message(`Add Camera Metadata`, message);
    disable_buttons(["camera_add_button"]);

    for (let input_id of ["sensor_width_input", "sensor_height_input", "focal_length_input", "image_width_px_input", "image_height_px_input"]) {
        $("#" + input_id).on("input", function(e) {
            if (update_sensor()) {
                enable_buttons(["camera_add_button"]);
            }
            else {
                disable_buttons(["camera_add_button"]);
            }
        });
    }
}

function add_make_model_fields(left_col_width_px) {

    let message = 
        `<table>` +
        `<tr>` +
            `<td>` + 
                `<div class="table_head" style="width: ${left_col_width_px}; padding-right: 10px">Make</div>` +
            `</td>` +
            `<td>` +
                `<div style="width: 250px">` +
                    `<input id="make_input" class="nonfixed_input">` +
                `</div>` +
            `</td>` +
        `</tr>` + 
        `<tr>` +
            `<td>` + 
                `<div class="table_head" style="width: ${left_col_width_px}; padding-right: 10px">Model</div>` +
            `</td>` +
            `<td>` +
                `<div style="width: 250px">` +
                    `<input id="model_input" class="nonfixed_input">` +
                `</div>` +
            `</td>` +
        `</tr>` +
        `</table>`;

    return message;
}

function add_make_model_metadata() {

    $("#modal_header_text").html("Add Camera Metadata");
    $("#modal_message").empty();

    let message = add_make_model_fields("60px");

    message = message + `<div style="height: 20px"></div>`;
    message = message + `<table><tr>` +
    `<td>` +
    `<button class="button-black button-black-hover" style="width: 220px; height: 30px;" id="camera_search_button" onclick="search_for_camera()">`+
        `Search For Camera</button>` +
    `</td>` +
    `</tr></table>`;

    show_modal_message(`Add Camera Metadata`, message);
    disable_buttons(["camera_search_button"]);
    for (let input of ["make_input", "model_input"]) {
        $("#" + input).on("input", function(e) {
            if (update_make_model()) {
                enable_buttons(["camera_search_button"]);
            }
            else {
                disable_buttons(["camera_search_button"]);
            }
        });
    }
}


function update_camera() {
    
    let make = $("#make_input").val();
    let model = $("#model_input").val();

    add_camera(make, model);

}
function add_camera(make, model) {

    let sensor_width = $("#sensor_width_input").val();
    let sensor_height = $("#sensor_height_input").val();
    let focal_length = $("#focal_length_input").val();
    let image_width_px = $("#image_width_px_input").val();
    let image_height_px = $("#image_height_px_input").val();

    let farm_name = $("#farm_combo").val();
    let field_name = $("#field_combo").val();
    let mission_date = $("#mission_combo").val();

    $.post($(location).attr('href'),
    {
        action: "add_camera",
        make: make,
        model: model,
        sensor_width: sensor_width,
        sensor_height: sensor_height,
        focal_length: focal_length,
        image_width_px: image_width_px,
        image_height_px: image_height_px,
        farm_name: farm_name,
        field_name: field_name,
        mission_date: mission_date
    },
    function(response, status) {

        if (response.error) {
            show_modal_message(`Error`, response.message);
        }
        else {
            camera_specs = response.camera_specs;
            show_overview();
            show_modal_message(`Success!`, 
                `Success! The provided metadata has been successfully processed.<br><br>You may now close this window.`);
        }
    });

}


function search_for_camera() {
    
    let make = $("#make_input").val();
    let model = $("#model_input").val();

    if (make in camera_specs && model in camera_specs[make]) {

        let sensor_width = camera_specs[make][model]["sensor_width"];
        let sensor_height = camera_specs[make][model]["sensor_height"];
        let focal_length = camera_specs[make][model]["focal_length"];
        let image_width_px = camera_specs[make][model]["image_width_px"];
        let image_height_px = camera_specs[make][model]["image_height_px"];

        let farm_name = $("#farm_combo").val();
        let field_name = $("#field_combo").val();
        let mission_date = $("#mission_combo").val();

        $.post($(location).attr('href'),
        {
            action: "add_camera",
            make: make,
            model: model,
            sensor_width: sensor_width,
            sensor_height: sensor_height,
            focal_length: focal_length,
            image_width_px: image_width_px,
            image_height_px: image_height_px,
            farm_name: farm_name,
            field_name: field_name,
            mission_date: mission_date
        },
        function(response, status) {

            if (response.error) {
                show_modal_message(`Error`, response.message);
            }
            else {
                camera_specs = response.camera_specs;
                show_overview();
                show_modal_message(`Success!`, 
                 `Success! The provided make and model are known to the system.<br><br>You may now close this window.`);
            }
        });

    }
    else {
        add_sensor_metadata(make, model);
    }
}

function submit_camera_height_change() {
    let new_camera_height = $("#camera_height_input").val()
    $.post($(location).attr('href'),
    {
        action: "update_camera_height",
        farm_name: $("#farm_combo").val(),
        field_name: $("#field_combo").val(),
        mission_date: $("#mission_combo").val(),
        camera_height: new_camera_height
    },
    function(response, status) {

        if (response.error) {
            show_modal_message(`Error`, 
            `An error occurred while updating the camera height:<br>` + response.message);
        }
        else {
            metadata["camera_height"] = new_camera_height;
            show_overview();
            show_modal_message(`Success!`, `The camera height has been successfully updated.`);
        }
    });
}

function edit_image_set_metadata() {
    show_modal_message(`Edit Camera Height`, 
    `<table>` +
        `<tr>` +
            `<td>` + 
                `<div style="width: 150px; padding-right: 10px">Camera Height (m)</div>` +
            `</td>` +
            `<td>` +
                `<div style="width: 250px">` +
                    `<input id="camera_height_input" class="nonfixed_input">` +
                `</div>` +
            `</td>` +
        `</tr>` +
    `</table>` +
    `<div style="height: 20px"></div>` +
    `<table>` +
        `<tr>` +
            `<td>` +
                `<button class="button-black button-black-hover" style="width: 220px; height: 30px;" id="submit_camera_height_button" onclick="submit_camera_height_change()">`+
                `Submit</button>` +
            `</td>` +
        `</tr>` +
    `</table>`

    );

    let camera_height = metadata["camera_height"];
    $("#camera_height_input").val(camera_height);


    $("#camera_height_input").on("input", function(e) {
        
        let new_camera_height = $("#camera_height_input").val();
        if (new_camera_height.length == 0) {
            enable_buttons(["submit_camera_height_button"]);
        }
        else if (!(isNumeric(new_camera_height))) {
            disable_buttons(["submit_camera_height_button"]);
        }
        else {
            let new_camera_height_val = parseFloat(new_camera_height);
            if (new_camera_height_val < MIN_CAMERA_HEIGHT || new_camera_height_val > MAX_CAMERA_HEIGHT) {
                disable_buttons(["submit_camera_height_button"]);
            }
            else {
                enable_buttons(["submit_camera_height_button"]);
            }
        }
    });
}

function reset_to_initial_apperance() {
    $("#farm_combo").prop("selectedIndex", -1);
    $("#field_combo").prop("selectedIndex", -1);
    $("#mission_combo").prop("selectedIndex", -1);
    $("#image_set_container").empty();

    clear_form();
    clear_train_form();
    create_image_set_list();

}

function show_overview() {

    viewing["browse"] = "overview";

    let farm_name = $("#farm_combo").val();
    let field_name = $("#field_combo").val();
    let mission_date = $("#mission_combo").val();

    $("#tab_details").empty();
    $("#tab_details").append(`<div style="height: 100px"></div><div class="loader"></div>`);

    $.post($(location).attr('href'),
    {
        action: "get_overview_info",
        farm_name: farm_name,
        field_name: field_name,
        mission_date: mission_date
    },
    function(response, status) {
        if (response.error) {
            show_modal_message(`Error`, response.message);
        }
        else {
            let annotation_info = response.annotation_info;
            metadata = response.metadata;
            let is_public = metadata["is_public"] ? "Yes": "No";

            let label_width = "200px";
            let value_width = "200px";
        
            $("#tab_details").empty();
            $("#tab_details").append(`<div style="height: 70px"></div>`);
        
            $("#tab_details").append(`<table style="height: 500px; border: 1px solid white; border-radius: 25px;" id="image_set_table"></table>`);
        
            $("#image_set_table").append(`<tr>`+
                `<td>` +
                    `<div style="width: 550px;" id="left_section"></div>` +
                `</td>` +
                `<td>` +
                    `<div style="width: 550px;" id="right_section">` +
                        `<table id="right_table" style="font-size: 14px"></table>` +
                    `</div>` +
                `</td>` +
            `</tr>`);
        
            let make = metadata["camera_info"]["make"];
            let model = metadata["camera_info"]["model"];
        
            let camera_height = metadata["camera_height"];
        
            let is_georeferenced;
            if (metadata["missing"]["latitude"] || metadata["missing"]["longitude"]) {
                is_georeferenced = "No";
            }
            else {
                is_georeferenced = "Yes";
            }

            $("#right_table").append(`<tr><td id="summary_entry"></td></tr>`);
            $("#summary_entry").append(`<div class="header2" style="font-size: 16px; text-align: left; width: 250px">Summary</div>`);
            $("#summary_entry").append(`<table id="image_stats_table" style="font-size: 14px; border: 1px solid white; border-radius: 10px; padding: 4px"></table>`);
        
            $("#image_stats_table").append(`<tr>` +
                    `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Images</div></td>` +
                    `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${annotation_info["num_images"]}</div></td>` +
                    `</tr>`);

            $("#image_stats_table").append(`<tr>` +
                `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Annotations</div></td>` +
                `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${annotation_info["num_annotations"]}</div></td>` +
                `</tr>`);

            $("#image_stats_table").append(`<tr>` +
                `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Regions of Interest</div></td>` +
                `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${annotation_info["num_regions_of_interest"]}</div></td>` +
                `</tr>`);

            $("#image_stats_table").append(`<tr>` +
                `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Fine-Tuning Regions</div></td>` +
                `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${annotation_info["num_fine_tuning_regions"]}</div></td>` +
                `</tr>`);

            $("#image_stats_table").append(`<tr>` +
                `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Test Regions</div></td>` +
                `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${annotation_info["num_test_regions"]}</div></td>` +
                `</tr>`);

            $("#right_table").append(`<tr><td><div style="height: 25px"></div></td></tr>`);
            $("#right_table").append(`<tr><td id="image_set_metadata_entry"></td></tr>`);
            $("#image_set_metadata_entry").append(`<table>` +
                    `<tr>` +
                        `<td>` +
                            `<div class="header2" style="font-size: 16px; text-align: left; width: 250px">Image Set Metadata</div>` +
                        `</td>` +
                        `<td style="width: 100%"></td>` +
                        `<td>` +
                            `<button id="edit_image_set_metadata_button" class="button-green button-green-hover" style="padding: 1px; font-size: 14px; width: 50px">Edit</button>` +
                        `</td>` +
                    `</tr>` +
                `</table>`);
            $("#image_set_metadata_entry").append(`<table id="image_set_metadata_table" style="font-size: 14px; border: 1px solid white; border-radius: 10px; padding: 4px"></table>`);

            $("#edit_image_set_metadata_button").click(function() {
                edit_image_set_metadata();
            });


            $("#image_set_metadata_table").append(`<tr>` +
                    `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Public</div></td>` +
                    `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${is_public}</div></td>` +
                    `</tr>`);
            $("#image_set_metadata_table").append(`<tr>` +
                    `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Georeferenced</div></td>` +
                    `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${is_georeferenced}</div></td>` +
                    `</tr>`);
            $("#image_set_metadata_table").append(`<tr>` +
                    `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Camera Height (m)</div></td>` +
                    `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${camera_height}</div></td>` +
                    `</td>` + 
                    `</tr>`);


            $("#right_table").append(`<tr><td><div style="height: 25px"></div></td></tr>`);
            $("#right_table").append(`<tr><td id="camera_metadata_entry"></td></tr>`);
            $("#camera_metadata_entry").append(`<table>` +
                `<tr>` +
                    `<td>` +
                        `<div class="header2" style="font-size: 16px; text-align: left; width: 250px">Camera Metadata</div>` +
                    `</td>` +
                    `<td style="width: 100%"></td>` +
                    `<td>` +
                        `<button id="edit_camera_metadata_button" class="button-green button-green-hover" style="padding: 1px; font-size: 14px; width: 50px">Edit</button>` +
                    `</td>` +
                `</tr>` +
            `</table>`);

            $("#camera_metadata_entry").append(`<table id="camera_specs_table" style="font-size: 14px; border: 1px solid white; border-radius: 10px; padding: 4px"></table>`);
        
        
            if ((make === "") || (model === "")) {
                let no_metadata_width = "410px";
        
                $("#camera_specs_table").append(`<tr>` +
                `<td style="height: 20px"></td>` +
                `</tr>`);
        
        
                $("#camera_specs_table").append(`<tr>` +
                    `<td><div style="width: ${no_metadata_width}">Metadata could not be extracted.</div></td>` +
                    `</tr>`);

                $("#edit_camera_metadata_button").click(function() {
                    add_make_model_metadata();
                });

            }
            else {
        
                $("#camera_specs_table").append(`<tr>` +
                    `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Make</div></td>` +
                    `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${make}</div></td>` +
                `</tr>`);
                $("#camera_specs_table").append(`<tr>` +
                    `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Model</div></td>` +
                    `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${model}</div></td>` +
                `</tr>`);
        
        
                if (make in camera_specs && model in camera_specs[make]) {
        
                    let sensor_height = camera_specs[make][model]["sensor_height"].toString();
                    let sensor_width = camera_specs[make][model]["sensor_width"].toString();
                    let focal_length = camera_specs[make][model]["focal_length"].toString();
                    let image_width_px = camera_specs[make][model]["image_width_px"].toString();
                    let image_height_px = camera_specs[make][model]["image_height_px"].toString();

                    $("#camera_specs_table").append(`<tr>` +
                        `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Sensor Width (mm)</div></td>` +
                        `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${sensor_width}</div></td>` +
                    `</tr>`);
                    $("#camera_specs_table").append(`<tr>` +
                        `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Sensor Height (mm)</div></td>` +
                        `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${sensor_height}</div></td>` +
                    `</tr>`);
                    $("#camera_specs_table").append(`<tr>` +
                        `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Focal Length (mm)</div></td>` +
                        `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${focal_length}</div></td>` +
                    `</tr>`);
                    $("#camera_specs_table").append(`<tr>` +
                        `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Raw Image Width (pixels)</div></td>` +
                        `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${image_width_px}</div></td>` +
                    `</tr>`);
                    $("#camera_specs_table").append(`<tr>` +
                        `<td style="padding: 2px 0px"><div class="table_header2" style="width: ${label_width};">Raw Image Height (pixels)</div></td>` +
                        `<td><div style="text-align: left; width: ${value_width}; margin-left: 10px">${image_height_px}</div></td>` +
                    `</tr>`);

                    $("#edit_camera_metadata_button").click(function() {
                        edit_metadata(make, model);
                    });
        
        
                }
        
                else {
                    $("#edit_camera_metadata_button").click(function() {
                        add_sensor_metadata(make, model);
                    });
                }
            }
        
        
            $("#left_section").append(
                        `<table id="left_table">` +
                            `<tr>` +
                                `<td>` +
                                `<button class="button-green button-green-hover" style="width: 220px; height: 80px; border-radius: 100px" onclick="workspace_request()">`+
                                    `<span><i class="fa-regular fa-pen-to-square" style="margin-right: 12px"></i>Workspace</span></button>` +
                                `</td>` +
                            `</tr>` +
                        `</table>`);
            
        
            if (annotation_info["num_annotations"] == 0) {
                $("#left_table").append(
                    `<tr style="height: 80px">` +
                    `<td>` +
                    `<button class="button-red button-red-hover" style="width: 220px; height: 35px;" onclick="delete_request()">`+
                        `<i class="fa-regular fa-circle-xmark" style="margin-right:8px"></i>Delete Image Set</button>` +
                    `</td>` +
                    `</tr>`);
            
            }
        }
    });
}


function fetch_upload_status(farm_name, field_name, mission_date) {

    $.post($(location).attr('href'),
    {
        action: "fetch_upload_status",
        farm_name: farm_name,
        field_name: field_name,
        mission_date: mission_date,
    },
    function(response, status) {
        image_sets_data[farm_name][field_name][mission_date] = response.status;

        if ((farm_name === $("#farm_combo").val() && field_name === $("#field_combo").val()) 
                && mission_date == $("#mission_combo").val()) {
            show_image_set_details();   
        }
    });
}

function fetch_and_show_results() {

    $.post($(location).attr('href'),
    {
        action: "fetch_results",
        farm_name: $("#farm_combo").val(),
        field_name: $("#field_combo").val(),
        mission_date: $("#mission_combo").val(),
    },
    function(response, status) {
        if (response.error) {
            $("#tab_details").empty();
            show_modal_message(`Error`, `An error occurred while fetching the image set results.`);
        }
        else {
            show_results(response);
        }

    });
}


function delete_result_request(result_type, result_id) {


    show_modal_message(`Are you sure?`, `<div style="height: 30px">Are you sure you want to destroy this result?</div>` +
        `<div style="height: 20px"></div>` +
        `<div id="modal_button_container" style="text-align: center">` +
        `<button class="button-red button-red-hover" `+
        `style="width: 150px" onclick="confirmed_delete_result_request('${result_type}', '${result_id}')">Destroy</button>` +
        `<div style="display: inline-block; width: 10px"></div>` +
        `<button class="button-green button-green-hover" ` +
        `style="width: 150px" onclick="close_modal()">Cancel</button>` +
        `<div style="height: 20px" id="loader_container"></div>` +
        `</div>`
    );
}

function confirmed_delete_result_request(result_type, result_id) {

    let farm_name = $("#farm_combo").val();
    let field_name = $("#field_combo").val();
    let mission_date = $("#mission_combo").val();


    $("#loader_container").append(
        `<div class="loader"></div>`
    );


    $.post($(location).attr('href'),
    {
        action: "delete_result",
        farm_name: farm_name,
        field_name: field_name,
        mission_date: mission_date,
        result_type: result_type,
        result_id: result_id
    },
    function(response, status) {
        close_modal();

        if (response.error) {
            show_modal_message(`Error`, `An error occurred while deleting the result.`);
        }
    });
}

function view_comment(comment) {
    show_modal_message("Comment", comment);
}




function create_result_entry(result) {

    let result_name = result["results_name"];
    if (get_text_width(result_name, "normal 12px arial") > 600) {
        result_name = result_name.substring(0, 24) + " ... " + result_name.substring(result_name.length - 24);
    }

    let start_date = timestamp_to_date(result["request_time"]);
    let end_date, aborted_date;
    let completed = "end_time" in result && (!("aborted_time" in result));
    let aborted = "aborted_time" in result;
    
    let disp_end_date, disp_end_title;
    if (completed) {
        end_date = timestamp_to_date(result["end_time"]);
        disp_end_date = end_date;
        disp_end_title = "End Time";
    }
    else if (aborted) {
        aborted_date = timestamp_to_date(result["aborted_time"]);
        disp_end_date = aborted_date;
        disp_end_title = "Aborted Time";

    }
    else {
        disp_end_date = " ";
        disp_end_title = " ";
    }

    let destroy_button_container_id = "destroy_button_container_" + result["request_time"];
    let main_result_container_id = "main_result_container_" + result["request_time"];



    let result_overview_info = 
    `<table style="font-size: 14px">` +
        `<tr>` +
                `<td style="height: 18px; text-align: right">` +
                    `<div style="color: #ddccbb; font-weight: 400; width: 90px">Start Time</div>` +
                `</td>` + 
                `<td style="text-align: left; padding-left: 15px; width: 100%;">` +
                    `<div>${start_date}</div>` +
                `</td>` +
        `</tr>` +
        `<tr>` +
                `<td style="height: 18px; text-align: right">` +
                    `<div style="color: #ddccbb; font-weight: 400; width: 90px">${disp_end_title}</div>` +
                `</td>` + 
                `<td style="text-align: left; padding-left: 15px; width: 100%;">` +
                    `<div>${disp_end_date}</div>` +
                `</td>` + 
        `</tr>` +
    `</table>`;

    let template = 
        `<tr style="border-bottom: 1px solid #4c6645; height: 70px">` +
            `<td><div style="width: 25px"></div></td>` +   
            `<td>` +
                `<div class="object_entry" style="text-align: left; font-size: 12px; width: 610px; height: 50px; border-radius: 10px;">` +
                    `<div style="padding-left: 10px; padding-top: 10px">${result_name}</div>` +
                `</div>` +
            `</td>` +

            `<td style="width: 100%">` +
                `<div class="table_entry" style="text-align: left;">${result_overview_info}</div>` +
            `</td>` +
            `<td id="${main_result_container_id}">` +

            `</td>` +
            `<td>` +
                `<div style="width: 60px"></div>` +
            `</td>` +
            `<td>` +
                `<table>` +
                    `<tr>` +
                        `<td>` +
                            `<div style="height: 24px">` +
                                `<button class="button-green button-green-hover" style="width: 160px; font-size: 14px; padding: 3px;" ` +
                                        `onclick="view_comment('${result["results_comment"]}')">` +

                                    `<i class="fa-solid fa-comment-dots" style="margin-right: 14px"></i><div style="display: inline-block; text-align: left; width: 100px">View Comment</div>` +
                                `</button>` +
                            `</div>` +
                        `</td>` +
                    `</tr>` +
                    `<tr>` +
                        `<td>` +
                            `<div style="height: 1px"></div>` +
                        `</td>` +
                    `</tr>` +
                    `<tr>` +
                        `<td>` +
                            `<div style="height: 24px" id="${destroy_button_container_id}">` +
                            `</div>` +
                        `</td>` +
                    `</tr>` +
                `</table>` +
            `</td>` +
            `<td>` +
                `<div style="width: 25px"></div>` +
            `</td>` +    
        `</tr>`;



    if (completed) {
        $("#completed_results_table").append(template);

        let farm_name = $("#farm_combo").val();
        let field_name = $("#field_combo").val();
        let mission_date = $("#mission_combo").val();

        let href = get_AC_PATH() + "/viewer/" + username + "/" +
                           farm_name + "/" + field_name + "/" + mission_date + "/" + result["result_uuid"];

        $("#" + main_result_container_id).append(
                `<button onclick=view_result('${href}') class="button-green button-green-hover" style="font-size: 16px; width: 190px; height: 50px; border-radius: 100px">` +
                    `<span>` +
                        `<i class="fa-regular fa-eye" style="margin-right:8px"></i>View Result` +
                    `</span>` +
                `</button>`
        );
        $("#" + destroy_button_container_id).append(
            `<button class="button-red button-red-hover" style="width: 160px; font-size: 14px; padding: 3px;" ` +
                    `onclick="delete_result_request('completed', '${result["result_uuid"]}')">` +
                `<i class="fa-regular fa-circle-xmark" style="margin-right: 14px"></i><div style="display: inline-block; text-align: left; width: 100px">Destroy Result</div>` +
            `</button>`
        );
    }
    else if (aborted) {
        $("#aborted_results_table").append(template);
        let view_error_message_button_id = "view_error_message_button_" + result["result_uuid"];

        $("#" + main_result_container_id).append(
            `<button class="button-green button-green-hover" style="font-size: 16px; width: 190px; height: 50px; border-radius: 100px" ` +
                `id="${view_error_message_button_id}">`+
                    `<i class="fa-solid fa-triangle-exclamation" style="margin-right:8px"></i>View Error Message` +
            `</button>`
        );
        $("#" + destroy_button_container_id).append(
            `<button class="button-red button-red-hover" style="width: 160px; font-size: 14px; padding: 3px;" ` +
                    `onclick="delete_result_request('aborted', '${result["result_uuid"]}')">` +
                `<i class="fa-regular fa-circle-xmark" style="margin-right: 14px"></i><div style="display: inline-block; text-align: left; width: 100px">Destroy Result</div>` +
            `</button>`
        );
        $("#" + view_error_message_button_id).click(function() {
            show_modal_message("Error Message", result["error_message"]);
        });


    }


    return template;

}

function view_result(href) {
    reset_to_initial_apperance();
    window.location.href = href;
}
function show_results(results) {

    viewing["browse"] = "results";


    completed_results = results.completed_results.sort(function(a, b) {
        return b["end_time"] - a["end_time"];
    });

    let completed_results_container_height = "400px";

    $("#tab_details").empty();

    $("#tab_details").append(`<div style="height: 40px"></div>`);
    $("#tab_details").append(`<div id="results_area" style="border-top: 1px solid white"></div>`);
    $("#results_area").append(
        `<ul class="nav" id="results_nav">` +
            `<li id="completed_results_tab_btn" class="nav">` +
                `<a class="nav"><span><i class="fa-solid fa-circle-check" style="margin-right: 3px"></i> Completed</span></a>` +
            `</li>` +
            `<li id="aborted_results_tab_btn" class="nav">` +
                `<a class="nav"><span><i class="fa-solid fa-circle-xmark" style="margin-right: 3px"></i> Aborted</span></a>` +
            `</li>` +
        `</ul>` +

        `<div id="completed_results" hidden>` +

            `<div style="height: 90px"></div>` +

            `<div style="width: 1450px; margin: 0 auto;">` +
                `<table>` +
                    `<tr>` +

                        `<td>` +
                            `<div style="width: 75px" class="header2">Sort By:</div>` +
                        `</td>` +
                        `<td>` +
                            `<select id="completed_results_sort_dropdown" class="nonfixed_dropdown" style="display: inline-block; width: 200px">` +
                                `<option value="end_time_desc" selected>Newest First</option>` +
                                `<option value="end_time_asc">Oldest First</option>` +
                                `<option value="result_name_desc">Result Name (DESC)</option>` +
                                `<option value="result_name_asc">Result Name (ASC)</option>` +
                            `</select>` +
                        `</td>` +
                        `<td style="width: 100%">` +
                        `</td>` +

                    `</tr>` +
                `</table>` +
            `</div>`+

            `<div class="scrollable_area" style="border-radius: 10px; height: ${completed_results_container_height}; width: 1450px; margin: 0 auto; overflow-y: scroll">` +
                `<table id="completed_results_table" style="border-collapse: collapse"></table>` +
            `</div>` +
        `</div>` +


        `<div id="aborted_results" hidden>` +
            `<div style="height: 90px"></div>` +
            `<div class="scrollable_area" style="border-radius: 10px; height: ${completed_results_container_height}; width: 1450px; margin: 0 auto; overflow-y: scroll">` +
                `<table id="aborted_results_table" style="border-collapse: collapse"></table>` +
            `</div>` +
        `</div>`

    );

    if (completed_results.length > 0) {
        for (let result of completed_results) {
            create_result_entry(result);
        }
    }
    else {
        $("#completed_results").empty();
        $("#completed_results").append(`<div style="height: 120px"></div>`);
        $("#completed_results").append(`<div>No Completed Results Found</div>`);
    }

    $("#completed_results_sort_dropdown").change(function() {
        let sort_val = $("#completed_results_sort_dropdown").val();
        let sorted_results;
        if ((sort_val === "end_time_desc") || (sort_val === "end_time_asc")) {
            sorted_results = completed_results.sort(function(a, b) {
                return b["end_time"] - a["end_time"];
            });
        }
        else if ((sort_val === "result_name_desc") || (sort_val === "result_name_asc")) {
            sorted_results = completed_results.sort(function(a, b) {
                return a.results_name.localeCompare(b.results_name);
            });
        }
        if ((sort_val === "end_time_asc") || (sort_val === "result_name_desc")) {
            sorted_results.reverse();
        }

        $("#completed_results_table").empty();
        for (let result of sorted_results) {
            create_result_entry(result);
        }

    });

    let aborted_results = results.aborted_results.sort(function(a, b) {
        return b["end_time"] - a["end_time"];
    });
    if (aborted_results.length > 0) {
        for (let result of aborted_results) {
            create_result_entry(result);
        }
    }
    else {
        $("#aborted_results").empty();
        $("#aborted_results").append(`<div style="height: 120px"></div>`);
        $("#aborted_results").append(`<div>No Aborted Results Found</div>`);
    }



    $("#completed_results_tab_btn").click(function() {
        active_results_tab_btn = "completed_results_tab_btn";
        show_results_tab();
    });

    $("#aborted_results_tab_btn").click(function() {
        active_results_tab_btn = "aborted_results_tab_btn";
        show_results_tab();
    });

    show_results_tab();

}

function show_image_set_details() {

    active_results_tab_btn = "completed_results_tab_btn";

    let farm_name = $("#farm_combo").val();
    let field_name = $("#field_combo").val();
    let mission_date = $("#mission_combo").val();


    $("#image_set_container").empty();


    let image_set_status = image_sets_data[farm_name][field_name][mission_date]["status"];
    if (image_set_status === "uploaded") {

        $("#image_set_container").append(`<ul class="nav" id="image_set_tabs"></ul>`);

        $("#image_set_tabs").append(
            `<li id="overview_tab_btn" class="nav tab-btn-active" onclick="show_image_set_tab(this.id)">` +
            `<a class="nav"><span><i class="fa-solid fa-bars" style="margin-right:3px"></i> Overview</span></a></li>`);
    
        $("#image_set_tabs").append(
            `<li id="results_tab_btn" class="nav" onclick="show_image_set_tab(this.id)">` +
            `<a class="nav"><span><i class="fa-solid fa-chart-column" style="margin-right:3px"></i> Results</span></a></li>`);
        $("#image_set_container").append(`<div id="tab_details"></div>`);

        show_image_set_tab("overview_tab_btn");
    }
    else if (image_set_status === "failed") {
        let error_message = image_sets_data[farm_name][field_name][mission_date]["error"];

        $("#image_set_container").append(`<div id="tab_details"></div>`);

        $("#tab_details").append(
        `<br><br><div>The following error occurred while processing the image set:</div><br><div>` + error_message + `</div><br>`);

        $("#tab_details").append(
            `<br><button class="button-red button-red-hover" style="width: 220px; height: 35px;" onclick="delete_request()">`+
            `<i class="fa-regular fa-circle-xmark" style="margin-right:8px"></i>Delete Image Set</button>`);

    }
    else {

        $("#image_set_container").append(`<div id="tab_details"></div>`);

        $("#tab_details").append(`<br><br><div class="loader"></div><br>` +
        `<div>This image set is currently being processed. ` +
        `This page will automatically update when the image set is ready to be viewed.<br></div>`);
    }
        
}


function initialize_browse() {


    $("#farm_combo").empty();
    $("#field_combo").empty();
    $("#mission_combo").empty();
    $("#image_set_container").empty();


    for (let farm_name of natsort(Object.keys(image_sets_data))) {
        $("#farm_combo").append($('<option>', {
            value: farm_name,
            text: farm_name
        }));
    }
    
    $("#farm_combo").change(function() {

        let farm_name = $(this).val();

        $("#field_combo").empty();
        $("#mission_combo").empty();
        $("#right_panel").empty();

        for (let field_name of natsort(Object.keys(image_sets_data[farm_name]))) {
            $("#field_combo").append($('<option>', {
                value: field_name,
                text: field_name
            }));
        }
        $("#field_combo").val($("#field_combo:first").val()).change();
    });


    $("#field_combo").change(function() {

        let farm_name = $("#farm_combo").val();
        let field_name = $(this).val();

        $("#mission_combo").empty();
        $("#right_panel").empty();

        for (let mission_date of natsort(Object.keys(image_sets_data[farm_name][field_name]))) {
            $("#mission_combo").append($('<option>', {
                value: mission_date,
                text: mission_date
            }));
        }
        $("#mission_combo").val($("#mission_combo:first").val()).change();
    });

    $("#farm_combo").prop("selectedIndex", -1);

}


$(document).ready(function() {


    image_sets_data = data["image_sets_data"];
    camera_specs = data["camera_specs"];
    objects = data["objects"];
    available_image_sets = data["available_image_sets"];
    overlay_appearance = data["overlay_appearance"];


    if (data["maintenance_time"] !== "") {
        $("#maintenance_message").html("Site maintenance is scheduled for " + data["maintenance_time"] + ".");
        $("#maintenance_message").show();
    }


    initialize_browse();
    initialize_upload();
    initialize_train();

    let socket = io(
    "", {
       path: get_AC_PATH() + "/socket.io"
    });

    socket.emit("join_home", username);

    socket.on("upload_change", function(message) {
        fetch_upload_status(message["farm_name"], message["field_name"], message["mission_date"]);
    });


    socket.on("results_change", function(message) {
        let farm_name = message["farm_name"];
        let field_name = message["field_name"];
        let mission_date = message["mission_date"];
        if ((farm_name === $("#farm_combo").val() && field_name === $("#field_combo").val()) 
            && mission_date == $("#mission_combo").val()) {
            if (viewing["browse"] === "results") {
                fetch_and_show_results();
            }
        }
    });

    socket.on("model_change", function() {

        if (viewing["train"] === "available") {
            show_available_train();
        }
        else if (viewing["train"] === "pending") {
            show_pending_train();
        }
        else if (viewing["train"] === "aborted") {
            show_aborted_train();
        }
    });

    $("#browse_tab_btn").click(function() {
        if (!global_disabled)
            show_tab("browse_tab_btn");
    });

    $("#upload_tab_btn").click(function() {
        if (!global_disabled)
            show_tab("upload_tab_btn");
    });

    $("#train_tab_btn").click(function() {
        if (!global_disabled)
            show_tab("train_tab_btn");
    });

    $("#mission_combo").change(function() {
        show_image_set_details();
    });

});
