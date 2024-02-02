let viewers = {};
let overlays = {};
let annotations = {};
let regions = {};
let cur_bounds = {};

let cur_inspected_set;
let added_image_sets = {};

const MODEL_NAME_FORMAT = /[\s `!@#$%^&*()+\=\[\]{}.;':"\\|,<>\/?~]/;


function clear_train_form() {

    added_image_sets = {};

    $("#model_name_input").val("");
    $('#model_public').prop('checked', true);

    $("#added_image_sets").empty();
    $("#model_class_list").empty();
    $("#added_filter_classes").empty();

    create_image_set_list();

    let first_row = document.getElementById("available_image_sets").rows[0]
    if (first_row != null) {
        first_row.scrollIntoView();
    }


    // TODO: this is a workaround to prevent the navigation tab from shrinking
    show_train_tab("available_train_tab_btn");
    show_train_tab("submit_train_tab_btn");

}


function show_train_tab(sel_tab_btn_id) {

    let image_set_tab_ids = [
        "submit_train_tab_btn",
        "available_train_tab_btn",
        "pending_train_tab_btn",
        "aborted_train_tab_btn"
    ];

    for (let tab_btn_id of image_set_tab_ids) {
        let tab_id = tab_btn_id.substring(0, tab_btn_id.length - 4);
        $("#" + tab_id).hide();
        $("#" + tab_btn_id).removeClass("tab-btn-active");
    }

    $("#" + sel_tab_btn_id).addClass("tab-btn-active");

    if (sel_tab_btn_id === "submit_train_tab_btn") {
        show_submit_train();
    }
    else if (sel_tab_btn_id === "available_train_tab_btn") {
        show_available_train();
    }
    else if (sel_tab_btn_id === "pending_train_tab_btn") {
        show_pending_train();
    }
    else {
        show_aborted_train();
    }
}




function box_intersects_region(box, region) {
    return ((box[1] < region[3] && box[3] > region[1]) && (box[0] < region[2] && box[2] > region[0]));
}

function get_num_useable_boxes(annotations) {

    let res = {};
    for (let image_name of Object.keys(annotations)) {
        for (let i = 0; i < annotations[image_name]["boxes"].length; i++) {
            let intersects = false;
            for (let j = 0; j < annotations[image_name]["fine_tuning_regions"].length; j++) {
                if (box_intersects_region(annotations[image_name]["boxes"][i], annotations[image_name]["fine_tuning_regions"][j])) {
                    intersects = true;
                    break;
                }
            }
            if (!(intersects)) {
                for (let j = 0; j < annotations[image_name]["test_regions"].length; j++) {
                    if (box_intersects_region(annotations[image_name]["boxes"][i], annotations[image_name]["test_regions"][j])) {
                        intersects = true;
                        break;
                    }
                }
            }
            if (intersects) {
                let class_idx = annotations[image_name]["classes"][i];
                if (!(class_idx in res)) {
                    res[class_idx] = 0;
                }
                res[class_idx]++;
            }
        }
    }
    return res;
}


function remove_filter_object_class(cls_ind) {

    let cur_obj_input = $("#object_filter_input").val();

    let row_id = "filter_cls_" + cls_ind;
    $("#" + row_id).remove();

    populate_object_filter_classes();

    $("#object_filter_input").val(cur_obj_input);
    if (cur_obj_input !== null) {
        enable_green_buttons(["add_filter_cls_button"]);
    }

    create_image_set_list();
}


function add_filter_cls_row(cls_name) {

    let cls_ind = (objects["object_names"]).indexOf(cls_name);
    let row_id = "filter_cls_" + cls_ind;

    $("#added_filter_classes").append(
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
                `<button onclick="remove_filter_object_class('${cls_ind}')"  style="width: 25px; height: 25px; border-radius: 5px; font-size: 12px;" class="button-red button-red-hover">` +
                    `<i class="fa-solid fa-circle-minus"></i>` + 
                `</button>` + 
            `</td>` +
            `<td><div style="width: 15px"></div></td>` +
        `</tr>` 
    );
}


function populate_object_filter_classes() {

    let added_filter_class_inds = [];
    $("#added_filter_classes tr").each(function() {
        let row_id_pieces = $(this).attr("id").split("_");
        let cls_ind = row_id_pieces[row_id_pieces.length-1];
        added_filter_class_inds.push(parseInt(cls_ind));
    });

    $("#object_filter_input").empty();

    if (added_filter_class_inds.length < 9) {
        for (let i = 0; i < objects["object_names"].length; i++) {
            if (!(added_filter_class_inds.includes(i))) {
                let object_name = objects["object_names"][i];
                $("#object_filter_input").append($('<option>', {
                    value: object_name,
                    text: object_name
                }));
            }
        }
    }

    $("#object_filter_input").prop("selectedIndex", -1);
    disable_green_buttons(["add_filter_cls_button"]);

}





function get_filtered_datasets() {
    let added_filter_classes = [];
    $("#added_filter_classes tr").each(function() {
        let id_pieces = $(this).attr("id").split("_");
        let class_ind = parseInt(id_pieces[id_pieces.length-1]);
        added_filter_classes.push(objects["object_names"][class_ind]);
    });

    let filtered_datasets = [];
    for (let username of Object.keys(available_image_sets)) {
        for (let farm_name of Object.keys(available_image_sets[username])) {
            for (let field_name of Object.keys(available_image_sets[username][farm_name])) {
                for (let mission_date of Object.keys(available_image_sets[username][farm_name][field_name])) {

                    let object_classes = available_image_sets[username][farm_name][field_name][mission_date]["object_classes"];

                    let valid = true;
                    if (added_filter_classes.length > 0) {
                        for (let object_class of added_filter_classes) {
                            if (!(object_classes.includes(object_class))) {
                                valid = false;
                                break;
                            }
                        }
                    }

                    if (valid) {
                        filtered_datasets.push({
                            "username": username,
                            "farm_name": farm_name,
                            "field_name": field_name,
                            "mission_date": mission_date,
                            "set_name": farm_name + " " + field_name + " " + mission_date,
                            "set_owner": username
                        });
                    }

                }
            }
        }
    }


    filtered_datasets.sort(function(a, b) {
        return a["set_owner"].localeCompare(b["set_owner"], undefined, {numeric: true, sensitivity: 'base'}) || 
               a["set_name"].localeCompare(b["set_name"], undefined, {numeric: true, sensitivity: 'base'});
    });

    return filtered_datasets;
}


function create_image_set_list() {


    let filtered_datasets = get_filtered_datasets();

    $("#available_image_sets").empty();

    for (let filtered_dataset of filtered_datasets) {
        let username = filtered_dataset["username"];
        let farm_name = filtered_dataset["farm_name"];
        let field_name = filtered_dataset["field_name"];
        let mission_date = filtered_dataset["mission_date"];
        let image_set_text_id = username + ":" + farm_name + ":" + field_name + ":" + mission_date;
        let image_set_details = create_image_set_details_table(username, farm_name, field_name, mission_date);

        let already_added = false;
        for (let added_image_set of Object.values(added_image_sets)) {
            if ((added_image_set["username"] === username && added_image_set["farm_name"] === farm_name) && 
                (added_image_set["field_name"] === field_name && added_image_set["mission_date"] === mission_date)) {
                
                already_added = true;
            }
        }


        let item;
        if (already_added) {
            item = `<div class="object_entry" style="padding: 4px; font-size: 14px; width: 100px">Added</div>`;
        }
        else {
            item =`<button onclick="inspect_image_set('${image_set_text_id}', false)" style="padding: 4px; font-size: 14px; width: 100px" class="button-green button-green-hover">Inspect</button>`;
        }
        

        $("#available_image_sets").append(
            `<tr style="border-bottom: 1px solid #4c6645">` + 
                `<td style="width: 100%">` +
                    `<div class="table_entry" style="text-align: left;">${image_set_details}</div>` +
                `</td>` +
                `<td>` +
                    item +
                `</td>` +
                `<td><div style="width: 5px;></div></td>` +
            `</tr>`
        );
    }

}



function create_viewer(id_prefix, dzi_image_paths) {

    $("#" + id_prefix + "_viewer").empty();


    viewers[id_prefix] = OpenSeadragon({
        id: id_prefix + "_viewer",
        sequenceMode: true,
        prefixUrl: get_AC_PATH() + "/osd/images/",
        tileSources: dzi_image_paths,
        showNavigator: false,
        maxZoomLevel: 1000,
        zoomPerClick: 1,
        nextButton: id_prefix + "_next",
        previousButton: id_prefix + "_prev",
        showNavigationControl: false,
    });



    overlays[id_prefix] = viewers[id_prefix].canvasOverlay({

        onOpen: function() {

            let region = regions[id_prefix][viewers[id_prefix].currentPage()];
            if (region != null) {

                let content_size = viewers[id_prefix].world.getItemAt(0).getContentSize();
                let image_w = content_size.x;
                let image_h = content_size.y;
                let hw_ratio = image_h / image_w;
                let viewport_bounds = [
                    region[1] / image_w,
                    (region[0] / image_h) * hw_ratio,
                    (region[3] - region[1]) / image_w,
                    ((region[2] - region[0]) / image_h) * hw_ratio
                ];
            
                cur_bounds[id_prefix] = new OpenSeadragon.Rect(
                    viewport_bounds[0],
                    viewport_bounds[1],
                    viewport_bounds[2],
                    viewport_bounds[3]
                );
            }
            else {
                cur_bounds[id_prefix] = null;
            }

        },
        onRedraw: function() {
            if (id_prefix in viewers) {
                let cur_tiles_url = viewers[id_prefix].source.tilesUrl;
                let basename_url = basename(cur_tiles_url);
                let cur_img_name = basename_url.substring(0, basename_url.length-6);
                let region = regions[id_prefix][viewers[id_prefix].currentPage()];
            
                let boxes_to_add = {};
                boxes_to_add["region_of_interest"] = {};
                boxes_to_add["region_of_interest"]["boxes"] = annotations[id_prefix][cur_img_name]["regions_of_interest"];
                boxes_to_add["fine_tuning_region"] = {};
                boxes_to_add["fine_tuning_region"]["boxes"] = annotations[id_prefix][cur_img_name]["fine_tuning_regions"];
                boxes_to_add["test_region"] = {};
                boxes_to_add["test_region"]["boxes"] = annotations[id_prefix][cur_img_name]["test_regions"]
                boxes_to_add["annotation"] = {};
                boxes_to_add["annotation"]["boxes"] = annotations[id_prefix][cur_img_name]["boxes"];
                boxes_to_add["annotation"]["classes"] = annotations[id_prefix][cur_img_name]["classes"];
                    
                let viewer_bounds = viewers[id_prefix].viewport.getBounds();

                let hw_ratio = overlays[id_prefix].imgHeight / overlays[id_prefix].imgWidth;
                let min_x = Math.floor(viewer_bounds.x * overlays[id_prefix].imgWidth);
                let min_y = Math.floor((viewer_bounds.y / hw_ratio) * overlays[id_prefix].imgHeight);
                let viewport_w = Math.ceil(viewer_bounds.width * overlays[id_prefix].imgWidth);
                let viewport_h = Math.ceil((viewer_bounds.height / hw_ratio) * overlays[id_prefix].imgHeight);
                let max_x = min_x + viewport_w;
                let max_y = min_y + viewport_h;


                if (region != null) {
                    min_y = Math.max(min_y, region[0]);
                    min_x = Math.max(min_x, region[1]);
                    max_y = Math.min(max_y, region[2]);
                    max_x = Math.min(max_x, region[3]);
                }
                

                let draw_order;
                if (region == null) {
                    draw_order = ["region_of_interest", "fine_tuning_region", "test_region", "annotation"];
                }
                else {
                    draw_order = ["annotation"];
                }
                for (let key of draw_order) { 

                    overlays[id_prefix].context2d().lineWidth = 2;

                    if (key === "region_of_interest") {
                        overlays[id_prefix].context2d().strokeStyle = overlay_appearance["colors"][key];
                        overlays[id_prefix].context2d().fillStyle = overlay_appearance["colors"][key] + "55";

                        for (let i = 0; i < boxes_to_add["region_of_interest"]["boxes"].length; i++) {

                            let region = boxes_to_add["region_of_interest"]["boxes"][i];
                            overlays[id_prefix].context2d().beginPath();
                            for (let j = 0; j < region.length; j++) {
                                let pt = region[j];
                    
                                let viewer_point = viewers[id_prefix].viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[1], pt[0]));
                                
                                if (j == 0) {
                                    overlays[id_prefix].context2d().moveTo(viewer_point.x, viewer_point.y);
                                }
                                else {
                                    overlays[id_prefix].context2d().lineTo(viewer_point.x, viewer_point.y);
                                }
                            }

                    
                            overlays[id_prefix].context2d().closePath();
                            overlays[id_prefix].context2d().stroke();
                            if (overlay_appearance["style"][key] == "fillRect") {
                                overlays[id_prefix].context2d().fill();
                            }
                    
                        }
                    }
                    else {


                        let visible_inds = [];
                        for (let i = 0; i < boxes_to_add[key]["boxes"].length; i++) {

                            let box = boxes_to_add[key]["boxes"][i];
                            let cls = boxes_to_add[key]["classes"][i];

                            if (((box[1] < max_x) && (box[3] > min_x)) && ((box[0] < max_y) && (box[2] > min_y))) {
                                if (key === "annotation") {
                                    if ($("#" + "image_set_class_" + cls.toString()).is(":checked")) {
                                        visible_inds.push(i);
                                    }
                                }
                                else {
                                    visible_inds.push(i);
                                }
                                
                                
                            }
                        }
                        if (visible_inds.length <= MAX_BOXES_DISPLAYED) {
                            for (let ind of visible_inds) {

                                let box = boxes_to_add[key]["boxes"][ind];
                                if (key === "annotation") {
                                    let cls = boxes_to_add[key]["classes"][ind];
                                    overlays[id_prefix].context2d().strokeStyle = overlay_appearance["colors"][key][cls];
                                    overlays[id_prefix].context2d().fillStyle = overlay_appearance["colors"][key][cls] + "55";
                                }
                                else {
                                    overlays[id_prefix].context2d().strokeStyle = overlay_appearance["colors"][key];
                                    overlays[id_prefix].context2d().fillStyle = overlay_appearance["colors"][key] + "55";
                                }



                                let viewer_point = viewers[id_prefix].viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[1], box[0]));
                                let viewer_point_2 = viewers[id_prefix].viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[3], box[2]));
                                
                                overlays[id_prefix].context2d().strokeRect(
                                    viewer_point.x,
                                    viewer_point.y,
                                    (viewer_point_2.x - viewer_point.x),
                                    (viewer_point_2.y - viewer_point.y)
                                );
                                //}
                                if (overlay_appearance["style"][key] == "fillRect") {
                                    overlays[id_prefix].context2d().fillRect(
                                        viewer_point.x,
                                        viewer_point.y,
                                        (viewer_point_2.x - viewer_point.x),
                                        (viewer_point_2.y - viewer_point.y)
                                    );
                                }
                            }
                        }
                    }
                }

                if (region != null) {

                    let image_px_width = overlays[id_prefix].imgWidth;
                    let image_px_height = overlays[id_prefix].imgHeight;
            
                    let inner_poly;
                    let outer_poly = [
                        [0-1e6, 0-1e6], 
                        [0-1e6, image_px_width+1e6], 
                        [image_px_height+1e6, image_px_width+1e6],
                        [image_px_height+1e6, 0-1e6]
                    ];

                    inner_poly = [
                        [region[0], region[1]],
                        [region[0], region[3]],
                        [region[2], region[3]],
                        [region[2], region[1]]
                    ];
            
                    overlays[id_prefix].context2d().fillStyle = "#222621";
                    overlays[id_prefix].context2d().beginPath();
            
                    for (let poly of [outer_poly, inner_poly]) {
            
                        for (let i = 0; i < poly.length+1; i++) {
                            let pt = poly[(i)%poly.length];
                            let viewer_point = viewers[id_prefix].viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[1], pt[0]));
            
                            if (i == 0) {
                                overlays[id_prefix].context2d().moveTo(viewer_point.x, viewer_point.y);
                            }
                            else {
                                overlays[id_prefix].context2d().lineTo(viewer_point.x, viewer_point.y);
                            }
                        }
                        overlays[id_prefix].context2d().closePath();
            
                    }
                    overlays[id_prefix].context2d().mozFillRule = "evenodd";
                    overlays[id_prefix].context2d().fill("evenodd");
                }

                if (cur_bounds[id_prefix] != null) {

                    if (region != null) {
            
                        viewers[id_prefix].world.getItemAt(0).setClip(
                            new OpenSeadragon.Rect(
                                region[1],
                                region[0],
                                (region[3] - region[1]),
                                (region[2] - region[0])
                            )
                        );
                    }
            
            
                    withFastOSDAnimation(viewers[id_prefix].viewport, function() {
                        viewers[id_prefix].viewport.fitBounds(cur_bounds[id_prefix]);
                    });
                    cur_bounds[id_prefix] = null;
                }
            }
        },
        clearBeforeRedraw: true
    });
}


function inspect_image_set(image_set_text_id, for_target) {

    let pieces = image_set_text_id.split(":");
    let username = pieces[0];
    let farm_name = pieces[1];
    let field_name = pieces[2];
    let mission_date = pieces[3];
    let image_set_dir = "usr/data/" + username + "/image_sets/" + farm_name + "/" + field_name + "/" + mission_date;
    let disp_text = username + " | " + farm_name + " | " + field_name + " | " + mission_date;

    $.post($(location).attr('href'),
    {
        action: "get_annotations",
        username: username,
        farm_name: farm_name,
        field_name: field_name,
        mission_date: mission_date
    },
    function(response, status) {

        if (response.error) {
            show_modal_message(`Error`, response.message);
        }
        else {
            let dzi_image_paths = [];
            let image_names;
            let id_prefix;
            let cur_regions;
            if (for_target) {
                id_prefix = "target";
                image_names = Object.keys(response.annotations);
                cur_regions = Array(image_names.length).fill(null);
            }
            else {
                id_prefix = "inspect";
                $("#inspected_set").html(disp_text);
                image_names = [];
                cur_regions = [];
                for (let image_name of Object.keys(response.annotations)) {
                    for (let i = 0; i < response.annotations[image_name]["fine_tuning_regions"].length; i++) {
                        image_names.push(image_name);
                        cur_regions.push(response.annotations[image_name]["fine_tuning_regions"][i]);
                    }
                    for (let i = 0; i < response.annotations[image_name]["test_regions"].length; i++) {
                        image_names.push(image_name);
                        cur_regions.push(response.annotations[image_name]["test_regions"][i]);
                    }
                }

            }


            $("#" + id_prefix + "_prev").show();
            $("#" + id_prefix + "_next").show();

            for (let image_name of image_names) {
                let dzi_image_path = get_AC_PATH() + "/" + image_set_dir + "/dzi_images/" + image_name + ".dzi";
                dzi_image_paths.push(dzi_image_path);
            }

            regions[id_prefix] = cur_regions;
            annotations[id_prefix] = response.annotations;

            let class_counts = get_num_useable_boxes(response.annotations);
            let object_classes = response.object_classes.split(",");


            cur_inspected_set = {
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "object_classes": object_classes
            };

            $("#inspection_area").show();

            if (!(id_prefix in viewers)) {
                create_viewer(id_prefix, dzi_image_paths);
            }
            else {
                viewers[id_prefix].tileSources = dzi_image_paths;
            }

            viewers[id_prefix].goToPage(0);

            $("#image_set_class_list").empty();
            let class_indices = Object.keys(class_counts).sort();
            for (let class_idx of class_indices) {
                let class_name = object_classes[class_idx];
                let image_set_class_id = "image_set_class_" + class_idx.toString();
                let bg_color = overlay_appearance["colors"]["annotation"][class_idx] + "aa";
                $("#image_set_class_list").append(
                    `<tr style="border-bottom: 1px solid #4c6645; height: 40px">` + 
                        `<td style="width: 100%">` +
                        `</td>` +
                        `<td>` +
                            `<div style="width: 180px" class="object_entry">${class_name}</div>` +
                        `</td>` +
                        `<td>` +
                            `<div style="width: 5px"></div>` +
                        `</td>` +
                        `<td>` +
                            `<label for="${image_set_class_id}" class="container" style="display: inline; margin-botton: 20px; margin-left: 12px">` +
                                `<input type="checkbox" id="${image_set_class_id}" name="${image_set_class_id}" checked></input>` +
                                `<span class="checkmark" style="background-color: ${bg_color}"></span>` +
                            `</label>` +
                        `</td>` +
                    `</tr>` 
                );

                $("#" + image_set_class_id).change(function() {
                    viewers[id_prefix].raiseEvent('update-viewport');
                });
            }

        }

    });
}


function show_submit_train() {
    viewing["train"] = "submit";
    $("#available_train_tab").hide();
    $("#pending_train_tab").hide();
    $("#aborted_train_tab").hide();
    $("#submit_train_tab").show();
    $("#submit_train_tab").show("fast", function() {
        if ("inspect" in viewers) {
            viewers["inspect"].viewport.goHome();
        }
    });
}



function show_pending_train() {
    viewing["train"] = "pending";
    $("#submit_train_tab").hide();
    $("#available_train_tab").hide();
    $("#aborted_train_tab").hide();
    $("#pending_train_tab").show();

    $("#pending_models_head").empty();
    $("#pending_models").empty();

    $.post($(location).attr('href'),
    {
        action: "fetch_my_models",
        model_state: "pending"
    },
    function(response, status) {

        if (response.error) {
            show_modal_message(`Error`, response.message);
        }
        else {
            
            if (response.models.length == 0) {
                $("#pending_models").append(
                    `<div>No Pending Models Found</div>`
                );

            }
            else {

                let models = response.models.sort(function(a, b) {
                    return b["log"]["model_name"] - a["log"]["model_name"];
                });

                $("#pending_models").append(
                    `<div class="scrollable_area" style="border-radius: 10px; height: 550px; width: 1200px; margin: 0 auto; overflow-y: scroll">` +
                        `<table id="pending_models_table" style="border-collapse: collapse"></table>` +
                    `</div>`
                );

                for (let model of models) {
                    create_model_entry(model["log"], "pending");
                }
            }

        }
    });
}

function show_aborted_train() {
    viewing["train"] = "aborted";
    $("#submit_train_tab").hide();
    $("#available_train_tab").hide();
    $("#pending_train_tab").hide();
    $("#aborted_train_tab").show();

    $("#aborted_models_head").empty();
    $("#aborted_models").empty();


    $.post($(location).attr('href'),
    {
        action: "fetch_my_models",
        model_state: "aborted"
    },
    function(response, status) {

        if (response.error) {
            show_modal_message(`Error`, response.message);
        }
        else {
            
            if (response.models.length == 0) {
                $("#aborted_models").append(
                    `<div>No Aborted Models Found</div>`
                );

            }
            else {

                let models = response.models.sort(function(a, b) {
                    return b["log"]["model_name"] - a["log"]["model_name"];
                });

                $("#aborted_models").append(
                    `<div class="scrollable_area" style="border-radius: 10px; height: 550px; width: 1200px; margin: 0 auto; overflow-y: scroll">` +
                        `<table id="aborted_models_table" style="border-collapse: collapse"></table>` +
                    `</div>`
                );

                for (let model of models) {
                    create_model_entry(model["log"], "aborted");
                }
            }

        }
    });

}

function destroy_model_request(model_state, model_name) {


    show_modal_message(`Are you sure?`, `<div style="height: 30px">Are you sure you want to destroy this model?</div>` +
        `<div style="height: 20px"></div>` +
        `<div id="modal_button_container" style="text-align: center">` +
        `<button id="confirm_delete" class="button-red button-red-hover" `+
        `style="width: 150px" onclick="confirmed_model_destroy_request('${model_state}', '${model_name}')">Destroy</button>` +
        `<div style="display: inline-block; width: 10px"></div>` +
        `<button id="cancel_delete" class="button-green button-green-hover" ` +
        `style="width: 150px" onclick="close_modal()">Cancel</button>` +
        `<div style="height: 20px" id="loader_container"></div>` +
        `</div>`
    );

}

function confirmed_model_destroy_request(model_state, model_name) {

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
        action: "destroy_model",
        model_name: model_name,
        model_state: model_state
    },
    function(response, status) {

        if (response.error) {
            show_modal_message(`Error`, response.message);
            $("#modal_close").off('click').on('click', function() {
                close_modal();
            });
        }
        else {
            close_modal();
            if (viewing["train"] === "available") {
                show_available_train();
            }
            else if (viewing["train"] === "aborted") {
                show_aborted_train();
            }
        }
    });



}


function create_model_entry(model_log, model_status) {

    let model_name = model_log["model_name"];
    let is_public = capitalizeFirstLetter(model_log["public"]);

    let start_date;
    start_date = timestamp_to_date(model_log["submission_time"]);

    let disp_end_title; 
    let disp_end_date;
    if (model_status === "available") {
        disp_end_date = timestamp_to_date(model_log["training_end_time"]);
        disp_end_title = "End Time";
    }
    else if (model_status === "aborted") {
        disp_end_date = timestamp_to_date(model_log["aborted_time"]);
        disp_end_title = "Aborted Time";
    }
    else {
        disp_end_date = " ";
        disp_end_title = " ";
    }

    let row_uuid = uuidv4();
    let destroy_button_container_id = row_uuid + "_destroy_button_container";
    let error_button_container_id = row_uuid + "_error_button_container";

    let template = 
        `<tr style="border-bottom: 1px solid #4c6645; height: 70px">` +
            `<td><div style="width: 15px"></div></td>` + 
            `<td>` +
                `<table style="font-size: 14px">` +
                    `<tr>` +
                        `<td style="text-align: right">` +
                            `<div style="color: #ddccbb; font-weight: 400; width: 80px">Model Name</div>` +
                        `</td>` + 
                        `<td style="text-align: left; padding-left: 15px;">` +
                            `<div style="width: 150px">${model_name}</div>` +
                        `</td>` +
                    `</tr>` +
                        `<tr>` +
                        `<td style="text-align: right">` +
                            `<div style="color: #ddccbb; font-weight: 400; width: 80px">Public?</div>` +
                        `</td>` + 
                        `<td style="text-align: left; padding-left: 15px;">` +
                            `<div style="width: 150px">${is_public}</div>` +
                        `</td>` +
                    `</tr>` +
                `</table>` +
            `</td>` +
            `<td>` +
                `<div style="width: 5px"></div>` +
            `</td>` +
            `<td>` +
                `<table style="font-size: 14px">` +
                    `<tr>` +
                        `<td style="height: 18px; text-align: right">` +
                            `<div style="color: #ddccbb; font-weight: 400; width: 90px">Start Time</div>` +
                        `</td>` + 
                        `<td style="text-align: left; padding-left: 15px;">` +
                            `<div style="width: 140px">${start_date}</div>` +
                        `</td>` +
                    `</tr>` +
                    `<tr>` +
                        `<td style="height: 18px; text-align: right">` +
                            `<div style="color: #ddccbb; font-weight: 400; width: 90px">${disp_end_title}</div>` +
                        `</td>` + 
                        `<td style="text-align: left; padding-left: 15px;">` +
                            `<div style="width: 140px">${disp_end_date}</div>` +
                        `</td>` + 
                    `</tr>` +
                `</table>` +
            `<td style="width: 100%">` +
            `</td>` +
            `<td>` +
                `<div style="width: 15px"></div>` +
            `</td>` +
            `<td>` +
                `<table>` +
                    `<tr>` +
                        `<td>` +
                            `<div id="${error_button_container_id}">` +
                            `</div>` +
                        `</td>` +
                    `</tr>` +
                    `<tr>` +
                        `<td>` +
                            `<div id="${destroy_button_container_id}">` +
                            `</div>` +
                        `</td>` +
                    `</tr>` +
                `</table>` +

            `</td>` +

            `<td>` +
                `<div style="width: 15px"></div>` +
            `</td>` +
        `</tr>`;

    $("#" + model_status + "_models_table").append(template);
    if (model_status === "available") {
        
        $("#" + destroy_button_container_id).append(
            `<button class="button-red button-red-hover"` +
                `onclick="destroy_model_request('available', '${model_name}')"  style="width: 180px; font-size: 14px; padding: 3px;">` +
                `<i class="fa-regular fa-circle-xmark" style="margin-right: 14px"></i><div style="display: inline-block; text-align: left;">Destroy Model</div>` +        
            `</button>`
        );

    }
    else if (model_status === "aborted") {

        let view_error_message_button_id = row_uuid + "_view_error_message_button";

        $("#" + destroy_button_container_id).append(
            `<button class="button-red button-red-hover"` +
                `onclick="destroy_model_request('aborted', '${model_name}')"  style="width: 180px; font-size: 14px; padding: 3px;">` +//<i class="fa-regular fa-circle-xmark"></i></button>`


                `<i class="fa-regular fa-circle-xmark" style="margin-right: 14px"></i><div style="display: inline-block; text-align: left; width: 130px">Destroy Model</div>` +
        
            `</button>`
        );

        $("#" + error_button_container_id).append(

            `<button class="button-green button-green-hover"` +
                `id="${view_error_message_button_id}" style="width: 180px; font-size: 14px; padding: 3px;">` +
                `<i class="fa-solid fa-triangle-exclamation" style="margin-right: 14px"></i><div style="display: inline-block; text-align: left; width: 130px">View Error Message</div>` +
            `</button>`
        );

        $("#" + view_error_message_button_id).click(function() {
            show_modal_message("Error Message", model_log["error_message"]);
        });
    }
    else {
        $("#" + destroy_button_container_id).append(
            `<div style="width: 100px"><div class="loader"></div></div>`
        );
    }
}



function show_available_train() {
    viewing["train"] = "available";
    $("#submit_train_tab").hide();
    $("#pending_train_tab").hide();
    $("#aborted_train_tab").hide();
    $("#available_train_tab").show();

    $("#available_models").empty();


    $.post($(location).attr('href'),
    {
        action: "fetch_my_models",
        model_state: "available"
    },
    function(response, status) {

        if (response.error) {
            show_modal_message(`Error`, response.message);
        }
        else {
            
            if (response.models.length == 0) {
                $("#available_models").append(
                    `<div>No Available Models Found</div>`
                );

            }
            else {

                $("#available_models").append(
                    `<div class="scrollable_area" style="border-radius: 10px; height: 550px; width: 1200px; margin: 0 auto; overflow-y: scroll">` +
                        `<table id="available_models_table" style="border-collapse: collapse"></table>` +
                    `</div>`
                );

                let models = response.models.sort(function(a, b) {
                    return b["log"]["model_name"] - a["log"]["model_name"];
                });

                for (let model of models) {
                    create_model_entry(model["log"], "available");
                }
            }
        }
    });
}

function remove_image_set(row_id) {

    $("#" + row_id).remove();

    let class_counts = {};

    for (let added_image_set of Object.values(added_image_sets)) {
        for (let added_class_idx of added_image_set["added_class_indices"]) {
            let class_name = added_image_set["object_classes"][added_class_idx];
            if (!(class_name in class_counts)) {
                class_counts[class_name] = 0;
            }
            class_counts[class_name]++;
        }
    }

    let classes_to_remove = [];
    for (let added_class_idx of added_image_sets[row_id]["added_class_indices"]) {
        let class_name = added_image_sets[row_id]["object_classes"][added_class_idx];
        if (class_counts[class_name] == 1) {
            classes_to_remove.push(class_name);
        }
    }

    for (let class_to_remove of classes_to_remove) {
        let model_class_row_id = "model_class_" + (objects["object_names"]).indexOf(class_to_remove);
        $("#" + model_class_row_id).remove();
    }

    delete added_image_sets[row_id];
    create_image_set_list();
}

function initialize_train() {

    create_image_set_list();

    show_train_tab("submit_train_tab_btn");


    populate_object_filter_classes();

    $("#object_filter_input").change(function() {
        enable_green_buttons(["add_filter_cls_button"]);
    });

    $("#add_filter_cls_button").click(function() {
        let cls_name = $("#object_filter_input").val();
        add_filter_cls_row(cls_name);
        populate_object_filter_classes();
        create_image_set_list();
    });


    $("#add_image_set_button").click(function() {

        let added_class_indices = [];
        for (let obj_index of Object.keys(cur_inspected_set["object_classes"])) {
            let image_set_class_id = "image_set_class_" + (obj_index).toString();
            if ($("#" + image_set_class_id).is(":checked")) {
                added_class_indices.push(parseInt(obj_index));
            }
        }

        if (added_class_indices.length == 0) {
            show_modal_message(`Denied`, 
            `At least one class must be selected before the new image set can be added.`);
        }
        else {

            let row_id = "added_set_" + uuidv4();

            let username = cur_inspected_set["username"];
            let farm_name = cur_inspected_set["farm_name"];
            let field_name = cur_inspected_set["field_name"];
            let mission_date = cur_inspected_set["mission_date"];

            added_image_sets[row_id] = {
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date,
                "added_class_indices": added_class_indices,
                "object_classes": cur_inspected_set["object_classes"]
            };



            let current_model_classes = [];
            $("#model_class_list tr").each(function() {
                let row_id_pieces = $(this).attr("id").split("_");
                let class_ind = parseInt(row_id_pieces[row_id_pieces.length-1]);
                let class_name = objects["object_names"][class_ind];
                current_model_classes.push(class_name);
            });
            let total_classes = current_model_classes.length;
            for (let added_class_idx of added_class_indices) {

                let class_name = cur_inspected_set["object_classes"][added_class_idx];
                if (!(current_model_classes.includes(class_name))) {
                    total_classes++;
                }
            }

            if (total_classes > 9) {
                show_modal_message(`Denied`, 
                `Training sets can have a maximum of nine distinct classes.` +
                ` The action cannot be performed because it would result in a training set with more than nine classes.`);
            }
            else {

                let image_set_details = create_image_set_details_table(username, farm_name, field_name, mission_date);

                $("#added_image_sets").append(
                    `<tr id="${row_id}" style="border-bottom: 1px solid #4c6645">` + 
                        `<td style="width: 100%">` +
                            `<div class="table_entry" style="text-align: left;">${image_set_details}</div>` +
                        `</td>` +

                        `<td>` +
                            `<button onclick="remove_image_set('${row_id}')" style="padding: 4px; font-size: 14px; width: 100px" class="button-red button-red-hover">Remove</button>` +
                        `</td>` +
                        `<td><div style="width: 5px;></div></td>` +
                    `</tr>`
                );


                for (let added_class_idx of added_class_indices) {

                    let class_name = cur_inspected_set["object_classes"][added_class_idx];
                    if (!(current_model_classes.includes(class_name))) {

                        let model_class_row_id = "model_class_" + (objects["object_names"]).indexOf(class_name);
                        $("#model_class_list").append(
                            `<tr id="${model_class_row_id}" style="border-bottom: 1px solid #4c6645; height: 40px">` + 
                                `<td style="width: 50%"></td>` +
                                `<td>` +
                                    `<div style="width: 180px" class="object_entry">${class_name}</div>` +
                                `</td>` +
                                `<td style="width: 50%"></td>` +
                            `</tr>` 
                        );
                    }
                }

                $("#inspection_area").hide();
                create_image_set_list();

            }
        }
    });

    $("#submit_training_request").click(function() {

        let num_image_sets = Object.keys(added_image_sets).length;
        if (num_image_sets == 0) {
            show_modal_message(`Error`, 
                `At least one training image set must be added.`);
            return;
        }


        let model_name = $("#model_name_input").val();
        let input_length = model_name.length;

        if (input_length == 0) {
            show_modal_message(`Error`,`A model name must be provided.`);
            return;            
        }
        if (input_length < 3) {
            show_modal_message(`Error`,`The provided model name is too short.`);
            return;
        }
        if (input_length > 50) {
            show_modal_message(`Error`,`The provided model name is too long.`);
            return;
        }

        if (MODEL_NAME_FORMAT.test(model_name)) {
            show_modal_message(`Error`,`The provided model contains illegal characters.`);
            return;
        }

        if (model_name.startsWith("random_weights")) {
            show_modal_message(`Error`,`The provided model name is not allowed.`);
            return;
        }

        let num_model_classes = $("#model_class_list tr").length;
        if (num_model_classes < 1 || num_model_classes > 9) {
            show_modal_message(`Error`,`The model's training set must have between one and nine classes.`);
            return;
        }


        let is_public = ($("#model_public").is(':checked')) ? "yes" : "no";

        $.post($(location).attr('href'),
        {
            action: "train",
            model_name: model_name,
            image_sets: JSON.stringify(added_image_sets),
            is_public: is_public
        },
    
        function(response, status) {

            if (response.error) {  
                clear_train_form();
                show_modal_message("Error", response.message);  
            }
            else {
                clear_train_form();
                show_modal_message("Success", response.message);
            }
        });


    });

}