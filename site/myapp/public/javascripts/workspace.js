let image_set_info;
let metadata;
let camera_specs;
let dzi_image_paths;
let annotations;
let image_to_dzi;
let predictions;
let overlay_appearance;
let hotkeys;
let tags;

let viewer;
let anno;
let overlay;
let prediction_anno;
let cur_img_name;
let cur_region_index;
let cur_nav_list;
let cur_view;

let show_bookmarks = true;

let cur_panel;
let cur_bounds = null;


let map_url = null;
let min_max_rec = null;

let model_unassigned = true;
let num_images_fully_trained_on;

let switch_model_data = {};


let selected_annotation_index = -1;
let selected_annotation = null;
let cur_edit_layer = "annotation";


let d_rgb;
let rgb_ctx;

let voronoi_data = {};

let gsd = null;
let cur_mouse_x;
let cur_mouse_y;


let cur_gridview_tiles;
let cur_gridview_tile_index;
let navigator_viewer;
let navigator_overview;
let grid_zoomed;


let keydown_handler = async function(e) {

    if (cur_view === "image") {

        if (e.key === hotkeys["Previous Image/Region"]) {
            change_to_prev_image();
        }
        else if (e.key === hotkeys["Next Image/Region"]) {
            change_to_next_image();
        }
        else if (e.key === hotkeys["Save Annotations"]) {
            $("#save_button").click();
        }
        else {

            let valid_keys = [];
            let key_mapping = {};
            for (let i = 1; i <= metadata["object_classes"].length; i++) {
                valid_keys.push(hotkeys["Class " + i]);
                key_mapping[hotkeys["Class " + i]] = i-1;
            }
            if ((cur_panel === "prediction") && (metadata["object_classes"].length > 1)) {
                valid_keys.push(hotkeys["All Classes"]);
                key_mapping[hotkeys["All Classes"]] = -1;
            }
            if (valid_keys.includes(e.key)) {

                let num_val = key_mapping[e.key];

                if (cur_panel === "annotation") {
                    $("#class_select").val(num_val).change();
                }
                else if (cur_panel === "prediction") {
                    $("#pred_class_select").val(num_val).change();
                }
            }
        }
    }
}


let selected_keydown_handler = async function(e) {

    if (e.key === hotkeys["Delete Annotation"]) {

        let selected = anno.getSelected();
        if (selected != null) {

            anno.removeAnnotation(selected);
            anno.cancelSelected();

            let sel_box_array;
            if (cur_edit_layer === "annotation") {
                sel_box_array = annotations[cur_img_name]["boxes"];
            }
            else if (cur_edit_layer === "region_of_interest") {
                sel_box_array = annotations[cur_img_name]["regions_of_interest"];
            }
            else if (cur_edit_layer === "fine_tuning_region") {
                sel_box_array = annotations[cur_img_name]["fine_tuning_regions"];
            }
            else {
                sel_box_array = annotations[cur_img_name]["test_regions"];
            }


            if (cur_edit_layer === "region_of_interest") {
                for (let tag_name of Object.keys(tags)) {
                    for (let nav_item of Object.keys(tags[tag_name])) {
                        let elements = nav_item.split("/");
                        let iter_region_index = parseInt(elements[1]);

                        let cur_affected_nav_item = cur_img_name + "/" + selected_annotation_index;
                        if (nav_item === cur_affected_nav_item) {
                            delete tags[tag_name][nav_item];
                        }
                        if (iter_region_index > selected_annotation_index) {
                            tags[tag_name][elements[0] + "/" + String(elements[1]-1)] = tags[tag_name][nav_item];
                            delete tags[tag_name][nav_item];
                        }
                    }
                    if (Object.entries(tags[tag_name]).length == 0) {
                        delete tags[tag_name];
                    }
                }
            }

            sel_box_array.splice(selected_annotation_index, 1);

            if (cur_edit_layer === "annotation") {
                annotations[cur_img_name]["classes"].splice(selected_annotation_index, 1);
                if (cur_img_name in voronoi_data && "annotation" in voronoi_data[cur_img_name]) {
                    delete voronoi_data[cur_img_name]["annotation"];
                }
            }
            selected_annotation = null;
            selected_annotation_index = -1;

            update_navigation_dropdown();

            if (cur_edit_layer === "annotation") {
                if (sel_box_array.length == 0) {
                    annotations[cur_img_name]["source"] = "NA";
                }
            }
            else {
                update_region_name();
                create_navigation_table();
            }

            $("#save_button").removeClass("button-green");
            $("#save_button").removeClass("button-green-hover");
            $("#save_button").addClass("button-red");
            $("#save_button").addClass("button-red-hover");
        }

    }
    else if (e.key === hotkeys["Save Annotations"]) {
        $("#save_button").click();
    }
    else {

        let valid_keys = [];
        let key_mapping = {};
        for (let i = 1; i <= metadata["object_classes"].length; i++) {
            valid_keys.push(hotkeys["Class " + i]);
            key_mapping[hotkeys["Class " + i]] = i-1;
        }


        if ((valid_keys.includes(e.key)) && (cur_edit_layer === "annotation")) {

            let held_annotation_index = selected_annotation_index;
            await anno.updateSelected(selected_annotation, true);

            let new_cls = key_mapping[e.key]; //parseInt(e.key) - 1;

            annotations[cur_img_name]["classes"][held_annotation_index] = new_cls;

            $("#save_button").removeClass("button-green");
            $("#save_button").removeClass("button-green-hover");
            $("#save_button").addClass("button-red");
            $("#save_button").addClass("button-red-hover");
            viewer.raiseEvent('update-viewport');
        }
    }
}



async function change_image(cur_nav_item) {

    await unselect_selected_annotation();

    document.getElementById(cur_nav_item + "_row").scrollIntoView({behavior: "instant"});

    let pieces = cur_nav_item.split("/");
    cur_img_name = pieces[0];
    cur_region_index = parseInt(pieces[1]);

    let index = cur_nav_list.findIndex(x => x == cur_nav_item);
    if (index == 0) {
        disable_green_buttons(["prev_image_button"]);
    }
    else {
        enable_green_buttons(["prev_image_button"]);
    }
    if (index == cur_nav_list.length - 1) {
        disable_green_buttons(["next_image_button"]);
    }
    else {
        enable_green_buttons(["next_image_button"]);
    }

    $("#image_name").text(cur_img_name);

    update_region_name();
    update_count_combo(false);

    set_count_chart_data();
    set_score_chart_data();
    update_score_chart();
    update_count_chart();

    if (cur_panel === "annotation") {
        show_annotation(true);
    }
    else if (cur_panel === "prediction") {
        show_prediction(true);
    }
    else {
        show_segmentation();
    }
}


function resize_poly_px_str(px_str) {
    
    let img_min_y;
    let img_min_x;
    let img_max_y;
    let img_max_x;

    let img_dims = viewer.world.getItemAt(0).getContentSize();
    img_min_y = 0;
    img_min_x = 0;
    img_max_y = img_dims.y;
    img_max_x = img_dims.x;


    let start = px_str.indexOf('"');
    let end = px_str.lastIndexOf('"');

    let coords_str = px_str.substring(start+1, end);

    let less_than_min_y = 0;
    let more_than_max_y = 0;
    let less_than_min_x = 0;
    let more_than_max_x = 0;

    let revised_coords = [];
    let lst_of_coord_strs = coords_str.split(" ");
    for (let coord_str of lst_of_coord_strs) {
        let coords = coord_str.split(",").map(x => parseFloat(x));

        if (coords[1] < img_min_y) {
            less_than_min_y++;
        }
        if (coords[1] > img_max_y) {
            more_than_max_y++;
        }
        if (coords[0] < img_min_x) {
            less_than_min_x++;
        }
        if (coords[0] > img_max_x) {
            more_than_max_x++;
        }
        revised_coords.push([(coords[0]), (coords[1])]);

    }
    let num_coords = revised_coords.length;
    if ((less_than_min_y == num_coords) ||
        (more_than_max_y == num_coords) ||
        (less_than_min_x == num_coords) ||
        (more_than_max_x == num_coords)) {
        
        return "illegal";
    }
    else {
        if (polygon_is_self_intersecting(revised_coords)) {
            return "illegal";
        }

        let clip_polygon = [[0, 0], [img_max_x, 0], [img_max_x, img_max_y], [0, img_max_y]];
        revised_coords = clip_polygons_xy(revised_coords, clip_polygon);
        let rounded_revised_coords = [];
        let added_str_coords = [];
        for (let j = 0; j < revised_coords.length; j++) {
            let coord = [(revised_coords[j][0]), (revised_coords[j][1])];
            let str_coord = JSON.stringify(coord);
            if (!(added_str_coords.includes(str_coord))) {
                rounded_revised_coords.push(coord);
                added_str_coords.push(str_coord);
            }
        }

        if (rounded_revised_coords.length == 0) {
            return "illegal";
        }

        if (get_polygon_area(rounded_revised_coords) < 5) {
            return "illegal";            
        }

        let revised_coords_str = `<svg><polygon points="${rounded_revised_coords.map(xy => xy.join(',')).join(' ')}"></polygon></svg>`
        return revised_coords_str;
    }
}





function resize_px_str(px_str) {
    px_str = px_str.substring(11);
    let px_lst = px_str.split(",").map(x => parseFloat(x));
    

    let img_min_y;
    let img_min_x;
    let img_max_y;
    let img_max_x;
    let navigation_type = $("#navigation_dropdown").val();
    if (navigation_type === "images") {
        let img_dims = viewer.world.getItemAt(0).getContentSize();
        img_min_y = 0;
        img_min_x = 0;
        img_max_y = img_dims.y;
        img_max_x = img_dims.x;
    }
    else {
        let region = annotations[cur_img_name][navigation_type][cur_region_index];
        let poly_bbox = get_bounding_box_for_polygon(region);
        img_min_y = poly_bbox[0];
        img_min_x = poly_bbox[1];
        img_max_y = poly_bbox[2];
        img_max_x = poly_bbox[3];

    }

    let box_min_x = px_lst[0];
    let box_min_y = px_lst[1];
    let box_max_x = px_lst[0] + px_lst[2];
    let box_max_y = px_lst[1] + px_lst[3];


    if (navigation_type === "images") {
        if ((box_min_x < img_min_x && box_max_x < img_min_x) || 
            (box_min_x > img_max_x && box_max_x > img_max_x) ||
            (box_min_y < img_min_y && box_max_y < img_min_y) ||
            (box_min_y > img_max_y && box_max_y > img_max_y)) {

            return "illegal";
        }
    }
    else {
        let box_pts = [
            [box_min_y, box_min_x],
            [box_min_y, box_max_x],
            [box_max_y, box_max_x],
            [box_max_y, box_min_x]
        ];
        let region = annotations[cur_img_name][navigation_type][cur_region_index];
        for (let pt of box_pts) {
            if (!(point_is_inside_polygon(pt, region))) {
                return "illegal";
            }
        }
    }


    box_min_x = Math.max(box_min_x, img_min_x);
    box_min_y = Math.max(box_min_y, img_min_y);

    box_max_x = Math.min(box_max_x, img_max_x);
    box_max_y = Math.min(box_max_y, img_max_y);

    let box_w = box_max_x - box_min_x;
    let box_h = box_max_y - box_min_y;


    let min_dim = 1;
    let max_dim = 800;
    if (box_w < min_dim) {

        let tentative_box_min_x = Math.floor(box_min_x);
        let tentative_box_max_x = tentative_box_min_x + 1;
        if (tentative_box_min_x < img_min_x) {
            box_min_x = img_min_x;
            box_max_x = img_min_x + min_dim;
        }
        else if (tentative_box_max_x > img_max_x) {
            box_min_x = (img_max_x) - min_dim;
            box_max_x = img_max_x;
        }
        else {
            box_min_x = tentative_box_min_x;
            box_max_x = tentative_box_max_x;
        }
        
    }

    // if (cur_edit_layer === "annotation") { 
    //     if (box_w > max_dim) {

    //         let tentative_box_min_x = box_centre_x - Math.floor(max_dim / 2);
    //         let tentative_box_max_x = box_centre_x + Math.floor(max_dim / 2);
    //         if (max_dim > (img_max_x - img_min_x)) {
    //             box_min_x = img_min_x;
    //             box_max_x = img_max_x;
    //         }
    //         else if (tentative_box_min_x < img_min_x) {
    //             box_min_x = img_min_x;
    //             box_max_x = img_min_x + max_dim;
    //         }
    //         else if (tentative_box_max_x > img_max_x) {
    //             box_min_x = (img_max_x) - max_dim;
    //             box_max_x = img_max_x;
    //         }
    //         else {
    //             box_min_x = tentative_box_min_x;
    //             box_max_x = tentative_box_max_x;
    //         }
    //     }
    // }


    if (box_h < min_dim) {
        let tentative_box_min_y = Math.floor(box_min_y);
        let tentative_box_max_y = tentative_box_min_y + 1;
        if (tentative_box_min_y < img_min_y) {
            box_min_y = img_min_y;
            box_max_y = img_min_y + min_dim;
        }
        else if (tentative_box_max_y > img_max_y) {
            box_min_y = (img_max_y) - min_dim;
            box_max_y = img_max_y;
        }
        else {
            box_min_y = tentative_box_min_y;
            box_max_y = tentative_box_max_y;
        }
    }

    // if (cur_edit_layer === "annotation") { 
    //     if (box_h > max_dim) {

    //         let tentative_box_min_y = box_centre_y - Math.floor(max_dim / 2);
    //         let tentative_box_max_y = box_centre_y + Math.floor(max_dim / 2);
    //         if (max_dim > (img_max_y - img_min_y)) {
    //             box_min_y = img_min_y;
    //             box_max_y = img_max_y;
    //         }
    //         else if (tentative_box_min_y < img_min_y) {
    //             box_min_y = img_min_y;
    //             box_max_y = img_min_y + max_dim;
    //         }
    //         else if (tentative_box_max_y > img_max_y) {
    //             box_min_y = (img_max_y) - max_dim;
    //             box_max_y = img_max_y;
    //         }
    //         else {
    //             box_min_y = tentative_box_min_y;
    //             box_max_y = tentative_box_max_y;
    //         }
    //     }
    // }


    box_w = box_max_x - box_min_x;
    box_h = box_max_y - box_min_y;

    let updated_px_str = "xywh=pixel:" + box_min_x + "," + box_min_y +
                        "," + box_w + "," + box_h;

    
    return updated_px_str;


}


function create_anno() {

    anno = OpenSeadragon.Annotorious(viewer, {
        disableEditor: true,
        disableSelect: true,
        readOnly: true,
        formatter: formatter,
        hotkey: hotkeys["Create Annotation"]
    });

    Annotorious.BetterPolygon(anno);

    if (cur_edit_layer === "annotation") {
        anno.setDrawingTool("rect");
    }
    else {
        anno.setDrawingTool("polygon");
    }

    anno.on('cancelSelected', function(selection) {
        anno.updateSelected(selection, true);
    });

    anno.on('createAnnotation', function(annotation) {

        selected_annotation_index = -1
        selected_annotation = null;

        
        let px_str = annotation["target"]["selector"]["value"];

        let anno_item;
        let illegal_anno_item = false;

        if (cur_edit_layer === "annotation") {
            px_str = px_str.substring(11);
            let px_lst = px_str.split(",").map(x => parseFloat(x));

            if (px_lst[0] == -1) {
                illegal_anno_item = true;
            }
            else {

                anno_item = [
                    (px_lst[1]), 
                    (px_lst[0]), 
                    (px_lst[1] + px_lst[3]),
                    (px_lst[0] + px_lst[2])
                ];
            }
        }
        else {
            let start = px_str.indexOf('"');
            let end = px_str.lastIndexOf('"');

            let coords_str = px_str.substring(start+1, end);
            let lst_of_coord_strs = coords_str.split(" ");
            anno_item = [];
            for (let coord_str of lst_of_coord_strs) {
                let coords = coord_str.split(",").map(x => parseFloat(x));
                anno_item.push([(coords[1]), (coords[0])]);
            }
        }

        let sel_array;
        if (cur_edit_layer === "annotation") {
            sel_array = annotations[cur_img_name]["boxes"];
        }
        else if (cur_edit_layer == "region_of_interest") {
            sel_array = annotations[cur_img_name]["regions_of_interest"];
        }
        else if (cur_edit_layer === "fine_tuning_region") {
            sel_array = annotations[cur_img_name]["fine_tuning_regions"];
        }
        else {
            sel_array = annotations[cur_img_name]["test_regions"];
        }

        if (!(illegal_anno_item)) {
            let illegal_intersection = false;


            if (cur_edit_layer !== "annotation") {
                if (sel_array.length >= 99) {
                    illegal_intersection = true;
                }
                
                if (polygon_is_self_intersecting(anno_item)) {
                    illegal_intersection = true;
                }

                let avoid = null;
                if (cur_edit_layer === "fine_tuning_region") {
                    avoid = "test_regions";
                }
                else if (cur_edit_layer === "test_region") {
                    avoid = "fine_tuning_regions";
                }

                if (avoid) {

                    for (let i = 0; i < annotations[cur_img_name][avoid].length; i++) {

                        let reg = annotations[cur_img_name][avoid][i];
                        let clipped_polygon = clip_polygons_yx(anno_item, reg);
                        let poly_area = get_polygon_area(clipped_polygon);
                        if (poly_area > 0) {
                            illegal_intersection = true;
                        }

                    }
                }
            }

            if (!(illegal_intersection)) {
                sel_array.push(anno_item);
                if (cur_edit_layer === "annotation") {
                    annotations[cur_img_name]["classes"].push(parseInt($("#class_select").val()));
                }
                update_navigation_dropdown();

                if (cur_edit_layer === "annotation") {
                    if (annotations[cur_img_name]["source"] === "NA") {
                        annotations[cur_img_name]["source"] = "manually_annotated_from_scratch";
                    }
                    else if (annotations[cur_img_name]["source"] === "unmodified_model_predictions") {
                        annotations[cur_img_name]["source"] = "edited_model_predictions";
                    }
                    else if (annotations[cur_img_name]["source"] === "uploaded") {
                        annotations[cur_img_name]["source"] = "uploaded_and_edited";
                    }
                    if (cur_img_name in voronoi_data && "annotation" in voronoi_data[cur_img_name]) {
                        delete voronoi_data[cur_img_name]["annotation"];
                    }
                }
                else {
                    update_region_name();
                    create_navigation_table();
                }

                $("#save_button").removeClass("button-green");
                $("#save_button").removeClass("button-green-hover");
                $("#save_button").addClass("button-red");
                $("#save_button").addClass("button-red-hover");
            }
        }

        anno.clearAnnotations();
        viewer.raiseEvent('update-viewport');
    });

    anno.on('createSelection', async function(selection) {

        selection.target.source = window.location.href;
        
        selection.body = [{
            type: 'TextualBody',
            purpose: 'class',
            value: 'object'
        }];

        let px_str = selection.target.selector.value;
        let updated_px_str;
        if (cur_edit_layer === "annotation") {
            updated_px_str = resize_px_str(px_str);
        }
        else { 
            updated_px_str = resize_poly_px_str(px_str);
        }
        if (updated_px_str === "illegal") {
            anno.clearAnnotations();
        }
        else {

            selection.target.selector.value = updated_px_str;

            // Make sure to wait before saving!
            await anno.updateSelected(selection);
            anno.saveSelected();
        }

    });

    anno.on('updateAnnotation', async function(annotation, previous) {

        let px_str = annotation.target.selector.value;
        let updated_px_str;
        if (cur_edit_layer === "annotation") {
            updated_px_str = resize_px_str(px_str);
        }
        else { 
            updated_px_str = resize_poly_px_str(px_str);
        }

        if (updated_px_str !== "illegal") {

            annotation.target.selector.value = updated_px_str;

            let updated_anno_item;
            if (cur_edit_layer === "annotation") {
                updated_px_str = updated_px_str.substring(11);
                let px_lst = updated_px_str.split(",").map(x => parseFloat(x));

                updated_anno_item = [
                    (px_lst[1]), 
                    (px_lst[0]), 
                    (px_lst[1] + px_lst[3]),
                    (px_lst[0] + px_lst[2])
                ];
            }
            else {
                let start = updated_px_str.indexOf('"');
                let end = updated_px_str.lastIndexOf('"');
            
                let coords_str = updated_px_str.substring(start+1, end);
                let lst_of_coord_strs = coords_str.split(" ");
                updated_anno_item = [];
                for (let coord_str of lst_of_coord_strs) {
                    let coords = coord_str.split(",").map(x => parseFloat(x));
                    updated_anno_item.push([coords[1], coords[0]]);
                }
            }

            let sel_array;
            if (cur_edit_layer === "annotation") {
                sel_array = annotations[cur_img_name]["boxes"];
            }
            else if (cur_edit_layer === "region_of_interest") {
                sel_array = annotations[cur_img_name]["regions_of_interest"];
            }
            else if (cur_edit_layer === "fine_tuning_region") {
                sel_array = annotations[cur_img_name]["fine_tuning_regions"];
            }
            else {
                sel_array = annotations[cur_img_name]["test_regions"];
            }

            let illegal_intersection = false;

            if (cur_edit_layer !== "annotation") {
                if (sel_array.length >= 99) {
                    illegal_intersection = true;
                }
                
                if (polygon_is_self_intersecting(updated_anno_item)) {
                    illegal_intersection = true;
                }

                let avoid = null;
                if (cur_edit_layer === "fine_tuning_region") {
                    avoid = "test_regions";
                }
                else if (cur_edit_layer === "test_region") {
                    avoid = "fine_tuning_regions";
                }

                if (avoid) {

                    for (let i = 0; i < annotations[cur_img_name][avoid].length; i++) {

                        let reg = annotations[cur_img_name][avoid][i];
                        let clipped_polygon = clip_polygons_yx(updated_anno_item, reg);
                        if (get_polygon_area(clipped_polygon) > 0) {
                            illegal_intersection = true;
                        }

                    }
                }
            }
            if (!(illegal_intersection)) {
                let prev_anno_item = sel_array[selected_annotation_index];
                sel_array[selected_annotation_index] = updated_anno_item;

                if (!(arraysEqual(updated_anno_item, prev_anno_item))) {
                    $("#save_button").removeClass("button-green");
                    $("#save_button").removeClass("button-green-hover");
                    $("#save_button").addClass("button-red");
                    $("#save_button").addClass("button-red-hover");


                    if (cur_edit_layer === "annotation") {
                    
                        if (annotations[cur_img_name]["source"] === "unmodified_model_predictions") {
                            annotations[cur_img_name]["source"] = "edited_model_predictions";
                        }

                        if (cur_img_name in voronoi_data && "annotation" in voronoi_data[cur_img_name]) {
                            delete voronoi_data[cur_img_name]["annotation"];
                        }
                    }
                }
            }
        }

        selected_annotation_index = -1
        selected_annotation = null;
        anno.clearAnnotations();
        viewer.raiseEvent('update-viewport');
    });

}


function anno_and_pred_onRedraw() {

    let navigation_type = $("#navigation_dropdown").val();

    let cur_pred_cls_idx = $("#pred_class_select").val();

    let boxes_to_add = {};
    if ((cur_panel === "annotation") || (cur_panel === "prediction")) {
        boxes_to_add["region_of_interest"] = {};
        boxes_to_add["region_of_interest"]["boxes"] = annotations[cur_img_name]["regions_of_interest"];
        boxes_to_add["fine_tuning_region"] = {};
        boxes_to_add["fine_tuning_region"]["boxes"] = annotations[cur_img_name]["fine_tuning_regions"];
        boxes_to_add["test_region"] = {};
        boxes_to_add["test_region"]["boxes"] = annotations[cur_img_name]["test_regions"];
    }


    if ((cur_panel === "annotation") || (cur_panel === "prediction" && ($("#annotation").is(":checked")))) {
        boxes_to_add["annotation"] = {};
        boxes_to_add["annotation"]["boxes"] = annotations[cur_img_name]["boxes"];
        boxes_to_add["annotation"]["classes"] = annotations[cur_img_name]["classes"];
    }

    if (((cur_panel == "prediction") && (cur_img_name in predictions)) && ($("#prediction").is(":checked"))) {
        boxes_to_add["prediction"] = {};
        boxes_to_add["prediction"]["boxes"] = predictions[cur_img_name]["boxes"];
        boxes_to_add["prediction"]["scores"] = predictions[cur_img_name]["scores"];
        boxes_to_add["prediction"]["classes"] = predictions[cur_img_name]["classes"];
    }

    let slider_val = Number.parseFloat($("#confidence_slider").val());
        
    let viewer_bounds = viewer.viewport.getBounds();
    let container_size = viewer.viewport.getContainerSize();

    let hw_ratio = overlay.imgHeight / overlay.imgWidth;
    let min_x = Math.floor(viewer_bounds.x * overlay.imgWidth);
    let min_y = Math.floor((viewer_bounds.y / hw_ratio) * overlay.imgHeight);
    let viewport_w = Math.ceil(viewer_bounds.width * overlay.imgWidth);
    let viewport_h = Math.ceil((viewer_bounds.height / hw_ratio) * overlay.imgHeight);
    let max_x = min_x + viewport_w;
    let max_y = min_y + viewport_h;

    if (cur_region_index != -1) {

        let cur_region = annotations[cur_img_name][navigation_type][cur_region_index];
        cur_region = get_bounding_box_for_polygon(cur_region);

        min_y = Math.max(min_y, cur_region[0]);
        min_x = Math.max(min_x, cur_region[1]);
        max_y = Math.min(max_y, cur_region[2]);
        max_x = Math.min(max_x, cur_region[3]);
    }

    overlay.context2d().font = "14px arial";

    if ((cur_panel === "prediction") && (!($("#image_visible_switch").is(":checked")))) {
        let viewer_point_1 = viewer.viewport.imageToViewerElementCoordinates(
            new OpenSeadragon.Point(0, 0));
        let viewer_point_2 = viewer.viewport.imageToViewerElementCoordinates(
                new OpenSeadragon.Point(overlay.imgWidth, overlay.imgHeight));
                
        overlay.context2d().fillStyle = "#222621";         
        overlay.context2d().fillRect(
            viewer_point_1.x - 10,
            viewer_point_1.y - 10,
            (viewer_point_2.x - viewer_point_1.x) + 20,
            (viewer_point_2.y - viewer_point_1.y) + 20,
        );
    }


    let voronoi_keys = [];
    if ($("#voronoi_annotation").is(":checked")) {
        voronoi_keys.push("annotation");
    }
    if ($("#voronoi_prediction").is(":checked")) {
        voronoi_keys.push("prediction");
    }

    if (cur_panel === "prediction") {
        for (let key of voronoi_keys) {
            if (!(cur_img_name in voronoi_data)) {
                voronoi_data[cur_img_name] = {};
            }
            if (!(key in voronoi_data[cur_img_name])) {
                voronoi_data[cur_img_name][key] = compute_voronoi(key);
            }
            if (voronoi_data[cur_img_name][key][cur_pred_cls_idx] != null) {
                let visible_edges = [];
                for (let edge of voronoi_data[cur_img_name][key][cur_pred_cls_idx].edges) {

                    let line_box_min_x = Math.min(edge.va.x, edge.vb.x);
                    let line_box_min_y = Math.min(edge.va.y, edge.vb.y);
                    let line_box_max_x = Math.max(edge.va.x, edge.vb.x);
                    let line_box_max_y = Math.max(edge.va.y, edge.vb.y);
                    if (((line_box_min_x < max_x) && (line_box_max_x > min_x)) && ((line_box_min_y < max_y) && (line_box_max_y > min_y))) {
                        visible_edges.push(edge);
                    }
            
                }
                if (visible_edges.length <= MAX_EDGES_DISPLAYED) {
                    for (let edge of visible_edges) {
                        if (cur_pred_cls_idx == -1) {
                            color = "#222621";
                        }
                        else {
                            color = overlay_appearance["colors"][key][cur_pred_cls_idx];
                        }
                        overlay.context2d().strokeStyle = color;
                        overlay.context2d().lineWidth = 2;
                
                        let viewer_point_1 = viewer.viewport.imageToViewerElementCoordinates(
                            new OpenSeadragon.Point(edge.va.x, edge.va.y));
                        let viewer_point_2 = viewer.viewport.imageToViewerElementCoordinates(
                                new OpenSeadragon.Point(edge.vb.x, edge.vb.y));    
                
                        overlay.context2d().beginPath();
                        overlay.context2d().moveTo(viewer_point_1.x, viewer_point_1.y);
                        overlay.context2d().lineTo(viewer_point_2.x, viewer_point_2.y);
                        overlay.context2d().closePath();
                        overlay.context2d().stroke();
                    }
                }
            }
        }
    }

    let draw_order = overlay_appearance["draw_order"];
    for (let key of draw_order) {

        if (!(key in boxes_to_add)) {
            continue;
        }
        

        overlay.context2d().lineWidth = 2;

        if ((key === "region_of_interest" || key === "fine_tuning_region") || key === "test_region") {

            overlay.context2d().strokeStyle = overlay_appearance["colors"][key];
            overlay.context2d().fillStyle = overlay_appearance["colors"][key] + "55";


            for (let i = 0; i < boxes_to_add[key]["boxes"].length; i++) {

                if ((cur_edit_layer === key) && (i == selected_annotation_index)) {
                    continue;
                }
                let region = boxes_to_add[key]["boxes"][i];

                overlay.context2d().beginPath();
                for (let j = 0; j < region.length; j++) {
                    let pt = region[j];
        
                    let viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[1], pt[0]));
                    
                    if (j == 0) {
                        overlay.context2d().moveTo(viewer_point.x, viewer_point.y);
                    }
                    else {
                        overlay.context2d().lineTo(viewer_point.x, viewer_point.y);
                    }
                }

                overlay.context2d().closePath();
                overlay.context2d().stroke();
                if (overlay_appearance["style"][key] == "fillRect") {
                    overlay.context2d().fill();
                }
        
            }
        
        }
        else {

            let visible_inds = [];
            loop1:
            for (let i = 0; i < boxes_to_add[key]["boxes"].length; i++) {
                if ((cur_edit_layer === key) && (i == selected_annotation_index)) {
                    continue;
                }

                let box = boxes_to_add[key]["boxes"][i];
                if (key === "prediction") {
                    let score = boxes_to_add[key]["scores"][i];
                    if (score <= slider_val) {
                        continue;
                    }
                }
                if ((cur_panel === "prediction") && 
                    ((key === "annotation" || key === "prediction"))) {
                    if ((cur_pred_cls_idx != -1) && 
                        (boxes_to_add[key]["classes"][i] != cur_pred_cls_idx)) {
                        continue;
                    }
                }


                if (((box[1] < max_x) && (box[3] > min_x)) && ((box[0] < max_y) && (box[2] > min_y))) {

                    visible_inds.push(i);
                    if (visible_inds.length > MAX_BOXES_DISPLAYED) {
                        break loop1;
                    }
                }

            }

            if (visible_inds.length <= MAX_BOXES_DISPLAYED) {
                for (let ind of visible_inds) {
                    let box = boxes_to_add[key]["boxes"][ind];
                    let cls = boxes_to_add[key]["classes"][ind];
                    overlay.context2d().strokeStyle = overlay_appearance["colors"][key][cls];
                    overlay.context2d().fillStyle = overlay_appearance["colors"][key][cls] + "55";


                    let viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[1], box[0]));
                    let viewer_point_2 = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[3], box[2]));

                    overlay.context2d().strokeRect(
                        viewer_point.x,
                        viewer_point.y,
                        (viewer_point_2.x - viewer_point.x),
                        (viewer_point_2.y - viewer_point.y)
                    );

                    if (overlay_appearance["style"][key] == "fillRect") {
                        overlay.context2d().fillRect(
                            viewer_point.x,
                            viewer_point.y,
                            (viewer_point_2.x - viewer_point.x),
                            (viewer_point_2.y - viewer_point.y)
                        );
                    }
                }

                if ((key === "prediction") && ("prediction" in boxes_to_add) && ($("#scores_switch").is(":checked"))) {
                    for (let ind of visible_inds) {

                        let box = boxes_to_add[key]["boxes"][ind];
                        let score = boxes_to_add[key]["scores"][ind];
                        
                        let box_width_pct_of_image = (box[3] - box[1]) / overlay.imgWidth;
                        let disp_width = (box_width_pct_of_image / viewer_bounds.width) * container_size.x;
                        let box_height_pct_of_image = (box[3] - box[1]) / overlay.imgHeight;
                        let disp_height = (box_height_pct_of_image / viewer_bounds.height) * container_size.y;

                        if ((disp_width * disp_height) < 10) {
                            continue;
                        }

                        if (((box[1] < max_x) && (box[3] > min_x)) && ((box[0] < max_y) && (box[2] > min_y))) {

                            let viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[1], box[0]));
                            let score_text = (Math.ceil(score * 100) / 100).toFixed(2);

                            overlay.context2d().fillStyle = "white";
                            overlay.context2d().fillRect(
                                    viewer_point.x - 1,
                                    viewer_point.y - 20,
                                    36,
                                    20
                                );

                            overlay.context2d().fillStyle = "black";
                            overlay.context2d().fillText(score_text, 

                                viewer_point.x + 3,
                                viewer_point.y - 5
                            );
                        }
                    }
                }
            }
        }
    }

    if ((navigation_type === "regions_of_interest") || (navigation_type === "fine_tuning_regions" || navigation_type === "test_regions")) {
        let region = annotations[cur_img_name][navigation_type][cur_region_index];
        let image_px_width = metadata["images"][cur_img_name]["width_px"];
        let image_px_height = metadata["images"][cur_img_name]["height_px"];

        let inner_poly = region;
        let outer_poly = [
            [0-1e6, 0-1e6], 
            [0-1e6, image_px_width+1e6], 
            [image_px_height+1e6, image_px_width+1e6],
            [image_px_height+1e6, 0-1e6]
        ];

        overlay.context2d().fillStyle = "#222621";
        overlay.context2d().beginPath();

        for (let poly of [outer_poly, inner_poly]) {

            for (let i = 0; i < poly.length+1; i++) {
                let pt = poly[(i)%poly.length];
                let viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[1], pt[0]));

                if (i == 0) {
                    overlay.context2d().moveTo(viewer_point.x, viewer_point.y);
                }
                else {
                    overlay.context2d().lineTo(viewer_point.x, viewer_point.y);
                }
            }
            overlay.context2d().closePath();

        }
        overlay.context2d().mozFillRule = "evenodd";
        overlay.context2d().fill("evenodd");
    }
    if (cur_bounds != null) {

        if ((navigation_type === "regions_of_interest") || (navigation_type === "fine_tuning_regions" || navigation_type === "test_regions")) {

            let region = annotations[cur_img_name][navigation_type][cur_region_index];
            region = get_bounding_box_for_polygon(region);

            viewer.world.getItemAt(0).setClip(
                new OpenSeadragon.Rect(
                    region[1],
                    region[0],
                    (region[3] - region[1]),
                    (region[2] - region[0])
                )
            );
        }


        withFastOSDAnimation(viewer.viewport, function() {
            viewer.viewport.fitBounds(cur_bounds);
        });
        cur_bounds = null;
    }

    if (gsd != null) {
        let cur_zoom = viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true));
        let measure_width = Math.max(50, 0.08 * container_size.x);
        let measure_width_m = (gsd / cur_zoom) * measure_width;
        let unit;
        let measure_width_metric;
        if (measure_width_m < 1) {
            measure_width_metric = measure_width_m * 100;
            unit = "cm";
        }
        else {
            measure_width_metric = measure_width_m;
            unit = "m";
        }
        let measure_width_text = (Math.ceil(measure_width_metric * 100) / 100).toFixed(2) + " " + unit;


        overlay.context2d().fillStyle = "rgb(255, 255, 255, 0.7)";
        overlay.context2d().fillRect(
            container_size.x - measure_width - 20,
            container_size.y - 30,
            measure_width + 20,
            30
        );
        overlay.context2d().fillStyle = "black";
        overlay.context2d().fillRect(
            container_size.x - measure_width - 10,
            container_size.y - 8,
            measure_width,
            2
        );
        overlay.context2d().fillRect(
            container_size.x - measure_width - 10,
            container_size.y - 10,
            1,
            4
        );
        overlay.context2d().fillRect(
            container_size.x - 10,
            container_size.y - 10,
            1,
            4
        );

        overlay.context2d().fillText(measure_width_text, 
            container_size.x - measure_width - 10,
            container_size.y - 15
        );
    }

    if (cur_panel === "annotation") {
        if ((cur_mouse_x != null) && (cur_mouse_y != null)) {

            overlay.context2d().lineWidth = 2;
            if (cur_edit_layer === "annotation") {
                let cur_cls_idx = $("#class_select").val();
                overlay.context2d().strokeStyle = overlay_appearance["colors"][cur_edit_layer][cur_cls_idx];
            }
            else {
                overlay.context2d().strokeStyle = overlay_appearance["colors"][cur_edit_layer];
            }
            overlay.context2d().beginPath();
            overlay.context2d().moveTo(0, cur_mouse_y);
            overlay.context2d().lineTo(overlay._containerWidth, cur_mouse_y);
            overlay.context2d().stroke();
            overlay.context2d().closePath();


            overlay.context2d().beginPath();
            overlay.context2d().moveTo(cur_mouse_x, 0);
            overlay.context2d().lineTo(cur_mouse_x, overlay._containerHeight);
            overlay.context2d().stroke();
            overlay.context2d().closePath();
        }
    }

    let zoom_level = viewer.viewport.getZoom(true);
    $("#zoom_level_setting").text(zoom_level.toFixed(2));
    if (zoom_level < 1.1) {
        disable_green_buttons(["engage_grid"]);
    }
    else {
        enable_green_buttons(["engage_grid"]);
    }

}


function create_viewer(viewer_id) {


    viewer = OpenSeadragon({
        id: viewer_id,
        sequenceMode: true,
        prefixUrl: get_AC_PATH() + "/osd/images/",
        tileSources: dzi_image_paths,
        showNavigator: false,
        maxZoomLevel: 1000,
        zoomPerClick: 1,
        nextButton: "next-btn",
        previousButton: "prev-btn",
        showNavigationControl: false,
        imageSmoothingEnabled: true,
    });

    viewer.innerTracker.keyDownHandler = null;
    viewer.innerTracker.keyPressHandler = null;
    viewer.innerTracker.keyHandler = null;

    overlay = viewer.canvasOverlay({
        clearBeforeRedraw: true
    });

    create_anno();


    $("#" + viewer_id).on("pointermove", function(event) {
        if (cur_panel === "annotation") {
            $("#" + viewer_id).css("cursor", "none");
            cur_mouse_x = event.offsetX;
            cur_mouse_y = event.offsetY;
            overlay.clear();
            if ($("#engaged_grid_controls").is(":visible")) {
                gridview_onRedraw();
            }
            else {
                anno_and_pred_onRedraw();
            }
        }
        else {
            $("#" + viewer_id).css("cursor", "default");
        }
    });
    
    viewer.addHandler('canvas-click', function(event) {

        if (!(anno.readOnly)) {

            if (selected_annotation_index == -1) {

                let annotation_uuid = null;

                let webPoint = event.position;

                let viewportPoint = viewer.viewport.pointFromPixel(webPoint);
                let imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);

                selected_annotation_index = -1;

                let inside_box = false;
                let candidate_box_areas = [];
                let candidate_box_indices = [];
                let sel_box_array;
                if (cur_edit_layer === "annotation") {
                    sel_box_array = annotations[cur_img_name]["boxes"];
                }
                else if (cur_edit_layer === "region_of_interest") {
                    sel_box_array = annotations[cur_img_name]["regions_of_interest"];
                }
                else if (cur_edit_layer === "fine_tuning_region") {
                    sel_box_array = annotations[cur_img_name]["fine_tuning_regions"];
                }
                else {
                    sel_box_array = annotations[cur_img_name]["test_regions"];
                }


                if (cur_edit_layer === "annotation") {

                    let cur_cls_idx = parseInt($("#class_select").val());
                    for (let i = 0; i < sel_box_array.length; i++) {

                        if (annotations[cur_img_name]["classes"][i] !== cur_cls_idx) {
                            continue;
                        }

                        let box = sel_box_array[i];
                        if ((imagePoint.x >= box[1] && imagePoint.x <= box[3]) && (imagePoint.y >= box[0] && imagePoint.y <= box[2])) {
                            

                            inside_box = true;

                            let box_area = (box[3] - box[1]) * (box[2] - box[0]);

                            candidate_box_areas.push(box_area);
                            candidate_box_indices.push(i);

                        }
                    }
                }
                else {
                    for (let i = 0; i < sel_box_array.length; i++) {
                        let poly = sel_box_array[i];
                        if (point_is_inside_polygon([imagePoint.y, imagePoint.x], poly)) {
                            inside_box = true;
                            let area = get_polygon_area(poly);
                            candidate_box_areas.push(area);
                            candidate_box_indices.push(i);
                        }
                    }
                }
                if (candidate_box_indices.length > 0) {
                    selected_annotation_index = candidate_box_indices[argMin(candidate_box_areas)];
                }

                if (inside_box) {
                    let box = sel_box_array[selected_annotation_index];

                    annotation_uuid = uuidv4();
                    let box_str;
                    let selector_type;
                    if (cur_edit_layer === "annotation") {
                        box_str = [box[1], box[0], (box[3] - box[1]), (box[2] - box[0])].join(",");
                        box_str = "xywh=pixel:" + box_str;
                        selector_type = "FragmentSelector";
                    }
                    else {
                        let pt_strs = [];
                        for (let i = 0; i < box.length; i++) {
                            pt_strs.push(box[i][1] + "," + box[i][0]);
                        }
                        box_str = `<svg><polygon points="` + pt_strs.join(" ") + `"></polygon></svg>`;
                        selector_type = "SvgSelector";
                    }
                    selected_annotation = {
                        "type": "Annotation",
                        "body": [
                            {
                                type: 'TextualBody',
                                purpose: 'class',
                                value: 'object'
                            }
                        ],
                        "target": {
                            "source": "",
                            "selector": {
                                "type": selector_type,
                                "conformsTo": "http://www.w3.org/TR/media-frags/",
                                "value": box_str
                            }
                        },
                        "@context": "http://www.w3.org/ns/anno.jsonld",
                        "id": annotation_uuid
                    };

                    let overlay_identifier = cur_edit_layer;
                    if (cur_edit_layer == "annotation") {
                        overlay_identifier += "_" + $("#class_select").val().toString();
                    }

                    selected_annotation["body"].push({"value": overlay_identifier, "purpose": "highlighting"});

                    anno.clearAnnotations();
                    anno.addAnnotation(selected_annotation);

                    delay(10).then(() => {
                        // sometimes the annotation is selected, but sometimes it isn't?? if not selected, select it now.
                        if (anno.getSelected() == null) {
                            anno.selectAnnotation(selected_annotation);
                        }
                    });
                }
                viewer.raiseEvent('update-viewport');
            }
            else {

                let cur_selected = anno.getSelected();
                if (cur_selected == null) {
                    let webPoint = event.position;
                    let viewportPoint = viewer.viewport.pointFromPixel(webPoint);
                    let imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);

                    let px_str = selected_annotation.target.selector.value;
                    px_str = px_str.substring(11);
                    let px_lst = px_str.split(",").map(x => parseFloat(x));
                    let box = [
                        Math.round(px_lst[1]), 
                        Math.round(px_lst[0]), 
                        Math.round(px_lst[1] + px_lst[3]), 
                        Math.round(px_lst[0] + px_lst[2])
                    ];

                    if ((imagePoint.x >= box[1] && imagePoint.x <= box[3]) && (imagePoint.y >= box[0] && imagePoint.y <= box[2])) {
                        anno.clearAnnotations();
                        anno.addAnnotation(selected_annotation);
                        delay(10).then(() => anno.selectAnnotation(selected_annotation));
                    }
                    else {
                        selected_annotation_index = -1;
                        selected_annotation = null;
                        anno.clearAnnotations();
                        viewer.raiseEvent('update-viewport');
    
                    }

                }
                else {
                    anno.updateSelected(selected_annotation, true);
                }

            }
        }

    });



}




function build_map() {
    disable_buttons(["build_map_button"]);
    $("#build_loader").show();

    let sel_class_idx = $("#map_builder_class_select").val();

    let sel_interpolation = $("input[type='radio'][name='interpolation']:checked").val();

    if (metadata["is_ortho"]) {
        map_chart_tile_size = $("#tile_size_slider").val();
    }
    else {
        map_chart_tile_size = "";
    }


    $.post($(location).attr('href'),
    {
        action: "build_map",
        class_index: sel_class_idx,
        interpolation: sel_interpolation,
        tile_size: map_chart_tile_size
    },
    
    function(response, status) {
        $("#build_loader").hide();
        enable_buttons(["build_map_button"]);

        if (response.error) {  
            show_modal_message("Error", "An error occurred during the generation of the density map.");  
        }
        else {

            let timestamp = new Date().getTime();   
            
            let base = get_AC_PATH() + "/usr/data/" + username + "/image_sets/" + image_set_info["farm_name"] + "/" + 
                    image_set_info["field_name"] + "/" + image_set_info["mission_date"] + "/maps/" + sel_interpolation;

            map_url = base + "_predicted_map.svg?t=" + timestamp;

            let min_max_rec_url = base + "_min_max_rec.json?t=" + timestamp;

            $.getJSON(min_max_rec_url, function(data) {
                min_max_rec = data;
                draw_map_chart();
            });
        }
    });


}

async function show_map() {

    await unselect_selected_annotation();

    cur_bounds = null;
    overlay.onOpen = function() {};
    overlay.onRedraw = function() {};
    viewer = null;
    $("#seadragon_viewer").empty();


    cur_view = "map";

    $("#view_button_text").empty();
    $("#view_button_text").append(
        `<i class="fa-solid fa-image" style="padding-right: 10px; color: white;"></i>Image View`);

    $("#image_view_container").hide();
    $("#map_view_container").show();

    let map_can_be_built = (Object.keys(predictions).length == Object.keys(annotations).length);


    if (map_can_be_built) {
        $("#insufficient_annotation_container").hide();
        $("#map_builder_controls_container").show();
    }
    else {
        $("#map_builder_controls_container").hide();
        $("#insufficient_annotation_container").show();
    }

    draw_map_chart();
}


function show_image(image_name) {
    cur_view = "image";

    $("#view_button_text").empty();
    $("#view_button_text").append(
        `<i class="fa-solid fa-location-dot" style="padding-right: 10px; color: white;"></i>Map View`);
    
    $("#map_view_container").hide();
    $("#image_view_container").show();



    change_image(image_name + "/" + cur_region_index);
}


function save_annotations(callback=null) {


    $("#save_button").hide();
    $("#fake_save_button").show();

    for (let image_name of Object.keys(annotations)) {

        for (let i = 0; i < annotations[image_name]["boxes"].length; i++) {
            box = annotations[image_name]["boxes"][i];
            annotations[image_name]["boxes"][i] = [
                Math.round(box[0]),
                Math.round(box[1]),
                Math.round(box[2]),
                Math.round(box[3])
            ];
        }

        for (let key of ["fine_tuning_regions", "test_regions", "regions_of_interest"]) {
            for (let i = 0; i < annotations[image_name][key].length; i++) {
                for (let j = 0; j < annotations[image_name][key][i].length; j++) {
                    annotations[image_name][key][i][j] = [
                        Math.round(annotations[image_name][key][i][j][0]), 
                        Math.round(annotations[image_name][key][i][j][1])
                    ];
                }
            }
        }
    }
    if ((cur_panel === "annotation" || cur_panel === "prediction")) {
        overlay.clear();
        if ($("#engaged_grid_controls").is(":visible")) {
            gridview_onRedraw();
        }
        else {
            anno_and_pred_onRedraw();
        }
    }

    $.post($(location).attr('href'),
    {
        action: "save_annotations",
        annotations: JSON.stringify(annotations),
        excess_green_record: JSON.stringify(excess_green_record),
        tags: JSON.stringify(tags),
        is_public: JSON.stringify(metadata["is_public"]),
        object_classes: metadata["object_classes"].join(",")
    },
    
    function(response, status) {
        if (response.error) {
            show_modal_message("Error", "An error occurred while saving: " + response.message);
        }
        else {
            
            $("#save_button").removeClass("button-red");
            $("#save_button").removeClass("button-red-hover");
            $("#save_button").addClass("button-green");
            $("#save_button").addClass("button-green-hover");

            $("#fake_save_button").hide();
            $("#save_button").show();

        }

        if (callback !== null) {
            callback();
        }
    });

}



function confirmed_use_predictions() {

    let navigation_type = $('#navigation_dropdown').val();

    let cur_class_ind = parseInt($("#pred_class_select").val());

    /* Delete old boxes */
    let region;
    if (navigation_type === "images") {

        for (let i = 0; i < annotations[cur_img_name]["boxes"].length; i++) {

            if (annotations[cur_img_name]["classes"][i] == cur_class_ind) {
                annotations[cur_img_name]["boxes"].splice(i, 1);
                annotations[cur_img_name]["classes"].splice(i, 1);
            }
        }
        region = null;
    }
    else {
        region = annotations[cur_img_name][navigation_type][cur_region_index];
        for (let i = 0; i < annotations[cur_img_name]["boxes"].length; i++) {
            
            let box = annotations[cur_img_name]["boxes"][i];
            if (annotations[cur_img_name]["classes"][i] == cur_class_ind) {
                // not correct, but hopefully close enough
                if (point_is_inside_polygon([box[0], box[1]], region) ||
                    point_is_inside_polygon([box[0], box[3]], region) ||
                    point_is_inside_polygon([box[2], box[3]], region) ||
                    point_is_inside_polygon([box[2], box[1]], region)
                ) {
                    annotations[cur_img_name]["boxes"].splice(i, 1);
                    annotations[cur_img_name]["classes"].splice(i, 1);
                }
            }
        }
    }

    /* Add new boxes */
    let slider_val = Number.parseFloat($("#confidence_slider").val()); //.toFixed(2);
    for (let i = 0; i < predictions[cur_img_name]["scores"].length; i++) {
        if (predictions[cur_img_name]["scores"][i] > slider_val) {
            let box = predictions[cur_img_name]["boxes"][i];
            if (navigation_type === "images") {
                annotations[cur_img_name]["boxes"].push(box);
                annotations[cur_img_name]["classes"].push(cur_class_ind);
            }
            else {
                if (point_is_inside_polygon([box[0], box[1]], region) ||
                    point_is_inside_polygon([box[0], box[3]], region) ||
                    point_is_inside_polygon([box[2], box[3]], region) ||
                    point_is_inside_polygon([box[2], box[1]], region)
                ) {
                    annotations[cur_img_name]["boxes"].push(box);
                    annotations[cur_img_name]["classes"].push(cur_class_ind);
                }
            }
        }
    }
    annotations[cur_img_name]["source"] = "unmodified_model_predictions";
    close_modal();
    $("#save_button").removeClass("button-green");
    $("#save_button").removeClass("button-green-hover");
    $("#save_button").addClass("button-red");
    $("#save_button").addClass("button-red-hover");
    show_annotation();
}


async function show_annotation(change_image=false) {

    await unselect_selected_annotation();

    let prev_panel = cur_panel;

    $("#show_annotation_button").addClass("tab-btn-active");
    $("#show_prediction_button").removeClass("tab-btn-active");
    $("#show_segmentation_button").removeClass("tab-btn-active");

    cur_panel = "annotation";
    $("#prediction_panel").hide();
    $("#segmentation_panel").hide();
    $("#annotation_panel").show();



    if (viewer == null) {
        $("#seadragon_viewer").empty();
        create_viewer("seadragon_viewer");
    }


    overlay.onOpen = function() {
        set_cur_bounds();
    };

    cur_mouse_x = null;
    cur_mouse_y = null;


    
    overlay.onRedraw = anno_and_pred_onRedraw;

    viewer.zoomPerScroll = 1.2;
    viewer.panHorizontal = true;
    viewer.panVertical = true;

    anno.readOnly = false;

    if (change_image || prev_panel == "segmentation") {
        let dzi_image_path = image_to_dzi[cur_img_name];
        viewer.open(dzi_image_path);      
    } 
    else {
        viewer.world.resetItems();
    }

    if (!($("#control_panel").is(":visible"))) {
        $("#gridview_button").click();
    }
}

async function unselect_selected_annotation() {
    if (selected_annotation != null) {
        
        let cur_selected = anno.getSelected();
        if (cur_selected == null) {
            anno.clearAnnotations();
            selected_annotation_index = -1;
            selected_annotation = null;
            viewer.raiseEvent('update-viewport');
        }
        else {
            await anno.updateSelected(selected_annotation, true);
        }
    }
}



async function show_prediction(change_image=false) {

    await unselect_selected_annotation();

    let prev_panel = cur_panel;

    $("#show_annotation_button").removeClass("tab-btn-active");
    $("#show_prediction_button").addClass("tab-btn-active");
    $("#show_segmentation_button").removeClass("tab-btn-active");
    
    cur_panel = "prediction";


    $("#annotation_panel").hide();
    $("#segmentation_panel").hide();
    $("#prediction_panel").show();

    if (viewer == null) {
        $("#seadragon_viewer").empty();
        create_viewer("seadragon_viewer");
    }

    overlay.onOpen = function() {
        set_cur_bounds();
    };

    overlay.onRedraw = anno_and_pred_onRedraw;

    $("#predictions_unavailable").hide();
    $("#predictions_available").hide();
    
    update_count_combo(false);
    set_count_chart_data();
    set_score_chart_data();
    update_score_chart();
    update_count_chart();
    


    if (cur_img_name in predictions) {
        $("#predictions_available").show();
    }
    else {
        $("#predictions_unavailable").show();
    }

    viewer.zoomPerScroll = 1.2;
    anno.readOnly = true;
    
    if (change_image || prev_panel == "segmentation") {
        let dzi_image_path = image_to_dzi[cur_img_name];
        viewer.open(dzi_image_path);      
    } 
    else {
        viewer.world.resetItems();
    }

    // potentially remove some elements, depending on the zoom level
    resize_window();

    
}


async function show_segmentation() {

    await unselect_selected_annotation();

    $("#show_annotation_button").removeClass("tab-btn-active");
    $("#show_prediction_button").removeClass("tab-btn-active");
    $("#show_segmentation_button").addClass("tab-btn-active");

    cur_panel = "segmentation";

    $("#annotation_panel").hide();
    $("#prediction_panel").hide();
    $("#segmentation_panel").show();

    let cur_exg_val = excess_green_record[cur_img_name];
    $("#threshold_slider_val").html(cur_exg_val.toFixed(2));
    $("#threshold_slider").val(cur_exg_val);

    update_apply_current_threshold_to_all_images_button();
    cur_bounds = null;
    $("#enable_pan_button").click();

}

function pan_viewport() {

    $("#segmentation_loader").hide();

    if (viewer == null) {
        $("#seadragon_viewer").empty();
        create_viewer("seadragon_viewer");
    }
    overlay.onRedraw = function() {};
    overlay.onOpen = function() {

        if (cur_bounds) {
            withFastOSDAnimation(viewer.viewport, function() {
                viewer.viewport.fitBounds(cur_bounds);
            });
        }
        else {
            let tiledImage = viewer.world.getItemAt(0);
            let viewer_width = $("#seadragon_viewer").width();
            let targetZoom = tiledImage.source.dimensions.x / viewer_width;
            viewer.viewport.zoomTo(targetZoom, null, true);
        }
    };

    anno.readOnly = true;
    viewer.zoomPerScroll = 1;

    let dzi_image_path = image_to_dzi[cur_img_name];
    viewer.open(dzi_image_path);
    
}

function segment_viewport() {

    if (viewer != null) {
        cur_bounds = viewer.viewport.getBounds();

        let img = viewer.drawer.canvas.toDataURL("image/png");
        
        let canvas = document.createElement("canvas");
        canvas.id = "my_canvas";

        rgb_ctx = canvas.getContext("2d");

        let container_width = $("#seadragon_viewer").width();
        let container_height = $("#seadragon_viewer").height();



        let rgb_image = new Image();
        rgb_image.src = img;

        rgb_image.onload = function() {

            rgb_ctx.canvas.width = container_width
            rgb_ctx.canvas.height = container_height;

            let w = rgb_ctx.canvas.width;
            let h = rgb_ctx.canvas.height;

            rgb_ctx.drawImage(rgb_image, 0, 0, w, h);      // Set image to Canvas context
            d_rgb = rgb_ctx.getImageData(0, 0, w, h);      // Get image Data from Canvas context

            draw_segmentation();
        }
    }
    else {
        draw_segmentation();
    }
    container_height = $("#seadragon_viewer").height();
}



function draw_segmentation() {

    
    let threshold = excess_green_record[cur_img_name];
    let num_foreground = 0;
    let non_zero = [];
    for (let i = 0; i < d_rgb.data.length; i += 4) {
        r_val = d_rgb.data[i] / 255;
        g_val = d_rgb.data[i+1] / 255;
        b_val = d_rgb.data[i+2] / 255;
        if ((r_val != 0 || g_val != 0) || b_val != 0) {
            non_zero.push({"r_val": r_val, "g_val": g_val, "b_val": b_val});
        }
        let exg_val = (2 * g_val) - r_val - b_val;

        let is_foreground = exg_val > threshold;
        d_rgb.data[i+3] = is_foreground ? 255 : 30;

        if (is_foreground) {
            num_foreground++;
        }
        
    }
    rgb_ctx.putImageData(d_rgb, 0, 0);

    overlay.onOpen = function() {};
    overlay.onRedraw = function() {};
    viewer = null;
    let canvas_container_height = $("#seadragon_viewer").height() + "px";

    $("#seadragon_viewer").empty();
    $("#seadragon_viewer").append(
        `<div id="canvas_container" style="height: ${canvas_container_height}">`+
        `</div>`

    );

    $("#canvas_container").append(rgb_ctx.canvas);
    container_width = $("#seadragon_viewer").width();
    container_height = $("#seadragon_viewer").height();

    delay(1).then(() => {
        $("#segmentation_loader").toggleClass('load-complete');
        $("#segmentation_checkmark").toggle();

        enable_green_buttons(["segment_button"]);
    });

}





function update_results_name_input() {

    let format = /[`!@#$%^&*()+\=\[\]{};':"\\|,<>\/?~]/;
    let inputs_to_check = ["results_name_input"];
    for (let input of inputs_to_check) {
        let input_length = ($("#" + input).val()).length;
        if ((input_length < 1) || (input_length > 50)) {
            return false;
        }

        if (format.test($("#" + input).val())) {
            return false;
        }
    }
    return true;
}



function update_results_comment_input() {

    let format = /[`!@#$%^&*\=\[\]{}|<>?~]/;
    let inputs_to_check = ["results_comment_input"];
    for (let input of inputs_to_check) {
        let input_length = ($("#" + input).val()).length;
        if ((input_length > 255)) {
            return false;
        }

        if (format.test($("#" + input).val())) {
            return false;
        }
    }
    return true;
}

function submit_result_request() {

    let at_least_one_region = false;
    for (let region_key of ["regions_of_interest", "fine_tuning_regions", "test_regions"]) {
        for (let image_name of Object.keys(annotations)) {
            if (annotations[image_name][region_key].length > 0) {
                at_least_one_region = true;
                break;
            }
        }
    }
    let full_image_label;
    if (metadata["is_ortho"]) {
        full_image_label = "Full Orthomosaic";
    }
    else {
        full_image_label = "All Images";
    }

    let left_col_width_px = "180px";
    show_modal_message("Submit Result Request", 
        `<div>Please confirm your request.` +
        ` Upon completion, your results will be preserved under this image set's` +
        ` <em>Results</em> tab (accessible from the home page).</div>` +
        `<div style="height: 30px"></div>` +
        `<table>` +
            `<tr>` +
                `<td>` + 
                    `<div class="table_head" style="width: ${left_col_width_px}; padding-right: 10px">Name</div>` +
                `</td>` +
                `<td>` +
                    `<div style="width: 330px">` +
                        `<input id="results_name_input" class="nonfixed_input" style="width: 100%" value="My Result">` +
                    `</div>` +
                `</td>` +
            `</tr>` +
            `<tr style="height: 5px">` +
            `</tr>` +
            `<tr>` +
                `<td>` + 
                    `<div class="table_head" style="width: ${left_col_width_px}; height: 85px; padding-right: 10px">Comment</div>` +
                `</td>` +
                `<td>` +
                    `<div style="width: 330px; height: 85px">` +
                        `<textarea id="results_comment_input" class="nonfixed_textarea" style="width: 100%" rows="4"></textarea>` +
                    `</div>` +
                `</td>` +
            `</tr>` + 
            `<tr id="results_region_radio_row">` +
                `<td>` +
                    `<div class="table_head" style="width: ${left_col_width_px}; height: 50px; padding-right: 10px">Prediction Target</div>` +
                `</td>` +
                `<td>` +
                    `<table style="border: 1px solid grey; width: 330px; height: 50px;" id="result_regions_radio_container">` +
                        `<tr>` +
                            `<td style="width: 50%"></td>` +
                            `<td>` +
                                `<label class="custom_radio_container" style="width: 160px; padding-left: 25px"> ${full_image_label}` +
                                    `<input type="radio" name="result_regions_radio" value="images" checked>` +
                                    `<span class="custom_radio"></span>` +
                                `</label>` +
                            `</td>` +
                            `<td><div style="width: 20px"></div></td>` +
                            `<td>` +
                                `<label class="custom_radio_container" style="width: 115px; padding-left: 25px"> All Regions` +
                                    `<input type="radio" name="result_regions_radio" value="regions">` +
                                    `<span class="custom_radio"></span>` +
                                `</label>` +
                            `</td>` +
                            `<td style="width: 50%"></td>` +
                        `</tr>` +
                    `</table>` +
                `</td>` +
            `</tr>` +
            `<tr>` +
                `<td>` +
                    `<div class="table_head" style="width: ${left_col_width_px}; padding-right: 10px">Calc. Veg. Coverage</div>` +
                `</td>` +
                `<td>` +
                    `<div style="width: 330px; text-align: left; padding-left: 2px">` +
                        `<label for="calc_veg_coverage" class="container" style="display: inline; margin-bottom: 20px;">` +
                            `<input type="checkbox" id="calc_veg_coverage" name="calc_vegetation_coverage" checked>` +
                            `<span class="checkmark"></span>` +
                        `</label>` +
                    `</div>` +
                `</div>` +
            `</tr>` +
        `</table>` + 
        `<div style="height: 30px"></div>` +
        `<div id="modal_button_container" style="text-align: center">` +
        `<button id="confirm_results_request_button" class="button-green button-green-hover" `+
        `style="width: 200px">Submit Request</button></div>`

        , 750);

    if (!(at_least_one_region)) {

        $("#results_region_radio_row").hide();
    }

    $("#confirm_results_request_button").click(function() {

        let results_request_callback = function() {

            let result_type_val = $('input[name=result_regions_radio]:checked').val();
            let result_regions_only = (result_type_val === "regions");
            let predict_on_images = !(result_regions_only);

            let res = get_image_list_and_region_list_for_predicting_on_all(predict_on_images);
            let image_list = res[0];
            let region_list = res[1];

            let calculate_vegetation_record = $("#calc_veg_coverage").is(':checked');


            if (region_list.length == 0) {
                show_modal_message(`Error`, `At least one region must exist before predictions for "all regions" can be requested.`);
            }

            submit_prediction_request_confirmed(image_list, region_list, true, result_regions_only, calculate_vegetation_record);

        }

        save_annotations(results_request_callback);
    });

    
    for (let input_id of ["results_name_input"]) {
        $("#" + input_id).on("input", function(e) {
            if (update_results_name_input() && update_results_comment_input()) {
                enable_green_buttons(["confirm_results_request_button"]);
            }
            else {
                disable_green_buttons(["confirm_results_request_button"]);
            }
        });
    }

    for (let input_id of ["results_comment_input"]) {
        $("#" + input_id).on("input", function(e) {
            if (update_results_name_input() && update_results_comment_input()) {
                enable_green_buttons(["confirm_results_request_button"]);
            }
            else {
                disable_green_buttons(["confirm_results_request_button"]);
            }
        });
    }


}

function show_fine_tuning_modal() {
    show_modal_message(`Submit Fine-Tuning Request`,
    
        `<table>` +
            `<tr>` +
                `<td>` +
                    `<div class="header2">Training Regime</div>` +
                `</td>` +
                `<td style="width: 10px"></td>` +
                `<td>` +
                    `<select id="training_regime_dropdown" class="nonfixed_dropdown" style="width: 230px">` +
                        `<option value="fixed_num_epochs">Fixed Number of Epochs</option>` +
                        `<option value="train_val_split">Monitor Validation Loss</option>` +
                    `</select>` +
                `</td>` +
            `</tr>` +
        `</table>` +
        `<div style="height: 10px"></div>` +
        `<div id="fixed_num_epochs_settings">` +
            `<div style="border: 1px solid white; border-radius: 10px; padding: 10px; margin: 10px 20px">` +
                `<table>` + 
                    `<tr>` + 
                        `<td>` +
                            `<div style="width: 510px; padding: 3px 0px">All data will be used for fine-tuning. No data will be used for validation.</div>` +
                        `</td>` +
                        `<td style="width: 100%"></td>` +
                    `</tr>` +
                `</table>` +
                `<table>` + 
                    `<tr>` + 
                        `<td>` +
                            `<div style="width: 200px">Training will terminate after</div>` +
                        `</td>` +
                        `<td>` +
                            `<input id="num_epochs_input" class="number_input" style="width: 60px" type="number" min="1" max="400" value="200" />` +
                        `</td>` +
                        `<td>` +
                            `<div style="width: 5px"></div>` +
                        `</td>` +
                        `<td>` +
                            `<div style="width: 60px">epochs.</div>` +
                        `</td>` +
                        `<td style="width: 100%"></td>` +
                    `</tr>` +
                `</table>` +
            `</div>` +
        `</div>` +
        `<div id="train_val_split_settings" hidden>` +
            `<div style="border: 1px solid white; border-radius: 10px; padding: 10px; margin: 10px 20px">` +
                `<table>` + 
                    `<tr>` + 
                        `<td>` +
                            `<div style="width: 180px">Training / validation split:</div>` +
                        `</td>` +
                        `<td style="width: 10px"></td>` + 
                        `<td>` +
                            `<input id="training_percent_input" class="number_input" style="width: 50px" type="number" min="50" max="95" value="80" step="5" />` +
                        `</td>` +
                        `<td style="width: 5px"></td>` + 
                        `<td>` +
                            `<div style="width: 50px" id="validation_percent">/ 20</div>` +
                        `</td>` +
                        `<td style="width: 100%"></td>` + 
                    `</tr>` +
                `</table>` +
                `<table>` +
                    `<tr>` + 
                        `<td>` +
                            `<div style="width: 110px">Terminate after</div>` +
                        `</td>` +
                        `<td style="width: 5px"></td>` + 
                        `<td>` +
                            `<input id="improvement_tolerance" class="number_input" style="width: 50px" type="number" min="1" max="50" value="10" step="1" />` +
                        `</td>` +
                        `<td style="width: 10px"></td>` + 
                        `<td>` +
                            `<div style="width: 300px">epochs without validation improvement.</div>` +
                        `</td>` +
                        `<td style="width: 100%"></td>` + 
                    `</tr>` +
                `</table>` +
            `</div>` +
        `</div>` +
        `<div style="height: 10px"></div>` +
        `<table>` +
            `<tr>` +
                `<td style="width: 50%"></td>` +
                `<td>` +
                    `<button onclick="submit_fine_tuning_request()" style="width: 180px" class="button-green button-green-hover">Submit Request</button>` +
                `</td>` +
                `<td style="width: 50%"></td>` +
            `</tr>` +
        `</table>`                
    );

    $("#training_percent_input").change(function() {
        let training_percent = $("#training_percent_input").val();
        let validation_percent = 100 - training_percent;
        $("#validation_percent").html("/ " + validation_percent);
    }) 


    $("#training_regime_dropdown").change(function() {
        let training_regime = $("#training_regime_dropdown").val();
        if (training_regime === "fixed_num_epochs") {
            $("#fixed_num_epochs_settings").show();
            $("#train_val_split_settings").hide();
        }
        else {
            $("#train_val_split_settings").show();
            $("#fixed_num_epochs_settings").hide();
        }
    });
}



function submit_fine_tuning_request() {
    disable_model_actions();
    close_modal();

    let num_fine_tuning_regions = 0;
    for (let image_name of Object.keys(annotations)) {
        num_fine_tuning_regions += annotations[image_name]["fine_tuning_regions"].length;
    }

    if (num_fine_tuning_regions == 0) {
        show_modal_message("Error", "The image set must contain at least one fine-tuning region before fine-tuning can be initiated.");
        enable_model_actions();
    }
    else {

        let callback = function() {
            let training_regime = $("#training_regime_dropdown").val();
            let req_data = {
                action: "fine_tune",
                training_regime: training_regime,
            };
            if (training_regime === "fixed_num_epochs") {
                num_epochs = $("#num_epochs_input").val();
                req_data["num_epochs"] = num_epochs;
            }
            else {
                req_data["training_percent"] = $("#training_percent_input").val();
                req_data["improvement_tolerance"] = $("#improvement_tolerance").val();
            }

            $.post($(location).attr("href"),
            req_data,
            function(response, status) {
                if (response.error) {
                    show_modal_message("Error", response.message);
                }
            });

        }

        save_annotations(callback);
    }


}

function submit_prediction_request_confirmed(image_list, region_list, save_result, result_regions_only, calculate_vegetation_coverage) {

    disable_model_actions();
    close_modal();

    $.post($(location).attr("href"),
    {
        action: "predict",
        image_names: JSON.stringify(image_list),
        regions: JSON.stringify(region_list),
        save_result: save_result,
        regions_only: result_regions_only,
        calculate_vegetation_coverage: calculate_vegetation_coverage,
        results_name: $("#results_name_input").val(),
        results_comment: $("#results_comment_input").val()
    },
    function(response, status) {

        if (response.error) {
            show_modal_message("Error", response.message);
        }
    });

}

function show_model_details(model_creator, model_name) {

    $.post($(location).attr("href"),
    {
        action: "inspect_model",
        model_creator: model_creator,
        model_name: model_name
    },
    function(response, status) {

        if (response.error) {
            show_modal_message("Error", "An error occurred while fetching the model details.");
        }

        else {

            switch_model_data["inspected_model_log"] = response.model_log;

            $("#model_select_back_button").show();
            $("#random_weights_button").hide();
        
            $("#model_info").empty();
            $("#model_info").append(`<table id="details_table"></table>`);
        
            let image_sets_col_width = "240px";
            let image_set_entry_width = "240px";
            let model_viewer_width = "450px";
            let target_viewer_width = "450px";
            let viewer_height = "390px";
            $("#details_table").append(
                `<tr>` +
                    `<td style="width: ${image_sets_col_width}; padding-left: 8px"><h class="header2">Model Image Sets</h></td>` +
                    `<td style="width: ${model_viewer_width}">` +
                    `<table>` +
                        `<tr>` +
                            `<td><h style="width: 180px" class="header2">Model Image Set</h></td>` +
                            `<td style="width: 100%"></td>` +
                            `<td>` +
                                `<button id="prev_ims_button" class="button-green button-green-hover" style="padding: 2px; font-size: 14px; width: 50px">` +
                                    `<i class="fa-solid fa-circle-chevron-left"></i>` +
                                `</button>` +
                            `</td>` +
                            `<td>` +
                                `<button id="next_ims_button" class="button-green button-green-hover" style="padding: 2px; font-size: 14px; width: 50px">` +
                                    `<i class="fa-solid fa-circle-chevron-right"></i>` +
                                `</button>` +
                            `</td>` +  
                        `<tr>` +
                    `</table>` +
                    `<td style="width: ${model_viewer_width}">` +
                    `<table>` +
                        `<tr>` +
                            `<td><h style="width: 180px" class="header2">Current Image Set</h></td>` +
                            `<td style="width: 100%"></td>` +
                            `<td>` +
                                `<button id="prev_cs_button" class="button-green button-green-hover" style="padding: 2px; font-size: 14px; width: 50px">` +
                                    `<i class="fa-solid fa-circle-chevron-left"></i>` +
                                `</button>` +
                            `</td>` +
                            `<td>` +
                                `<button id="next_cs_button" class="button-green button-green-hover" style="padding: 2px; font-size: 14px; width: 50px">` +
                                    `<i class="fa-solid fa-circle-chevron-right"></i>` +
                                `</button>` +
                            `</td>` +  
                        `<tr>` +
                    `</table>` +
                    `</tr>` +
                `</tr>`);
        
        
            $("#details_table").append(
                `<tr>` +
                    `<td>` +
                        `<div class="scrollable_area" style="height: ${viewer_height}; border: none; overflow-y: scroll">` +
                            `<table id="model_image_sets"></table>` +
                        `</div>` +
                    `</td>` +
                    `<td><div id="model_viewer" class="viewer" style="height: ${viewer_height}; width: ${model_viewer_width}"></div></td>` +
                    `<td><div id="target_viewer" class="viewer" style="height: ${viewer_height}; width: ${target_viewer_width}"></div></td>` +        `</tr>`
            );

        
            let current_image_set_viewer = OpenSeadragon({
                id: "target_viewer",
                sequenceMode: true,
                prefixUrl: get_AC_PATH() + "/osd/images/",
                tileSources: dzi_image_paths,
                showNavigator: false,
                maxZoomLevel: 1000,
                zoomPerClick: 1,
                nextButton: "next_cs_button",
                previousButton: "prev_cs_button",
                showNavigationControl: false,
            });


            current_image_set_overlay = current_image_set_viewer.canvasOverlay({

                onOpen: function() {
                },
                onRedraw: function() {
                    let cur_tiles_url = current_image_set_viewer.source.tilesUrl;
                    let basename_url = basename(cur_tiles_url);
                    let current_image_set_image_name = basename_url.substring(0, basename_url.length-6);
                
                    let boxes_to_add = {};
                    boxes_to_add["region_of_interest"] = {};
                    boxes_to_add["region_of_interest"]["boxes"] = annotations[current_image_set_image_name]["regions_of_interest"];
                    boxes_to_add["fine_tuning_region"] = {};
                    boxes_to_add["fine_tuning_region"]["boxes"] = annotations[current_image_set_image_name]["fine_tuning_regions"];
                    boxes_to_add["test_region"] = {};
                    boxes_to_add["test_region"]["boxes"] = annotations[current_image_set_image_name]["test_regions"]
                    boxes_to_add["annotation"] = {};
                    boxes_to_add["annotation"]["boxes"] = annotations[current_image_set_image_name]["boxes"];
                    boxes_to_add["annotation"]["classes"] = annotations[current_image_set_image_name]["classes"];
                        
                    let viewer_bounds = current_image_set_viewer.viewport.getBounds();

                    let hw_ratio = current_image_set_overlay.imgHeight / current_image_set_overlay.imgWidth;
                    let min_x = Math.floor(viewer_bounds.x * current_image_set_overlay.imgWidth);
                    let min_y = Math.floor((viewer_bounds.y / hw_ratio) * current_image_set_overlay.imgHeight);
                    let viewport_w = Math.ceil(viewer_bounds.width * current_image_set_overlay.imgWidth);
                    let viewport_h = Math.ceil((viewer_bounds.height / hw_ratio) * current_image_set_overlay.imgHeight);
                    let max_x = min_x + viewport_w;
                    let max_y = min_y + viewport_h;

                    let draw_order = ["region_of_interest", "fine_tuning_region", "test_region", "annotation"];
                    
                    for (let key of draw_order) { 
                        

                        current_image_set_overlay.context2d().lineWidth = 2;

                        if ((key === "region_of_interest" || key === "fine_tuning_region") || key === "test_region") {

                            current_image_set_overlay.context2d().strokeStyle = overlay_appearance["colors"][key];
                            current_image_set_overlay.context2d().fillStyle = overlay_appearance["colors"][key] + "55";                            

                            for (let i = 0; i < boxes_to_add[key]["boxes"].length; i++) {

                                let region = boxes_to_add[key]["boxes"][i];
                                current_image_set_overlay.context2d().beginPath();
                                for (let j = 0; j < region.length; j++) {
                                    let pt = region[j];
                        
                                    let viewer_point = current_image_set_viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[1], pt[0]));
                                    
                                    if (j == 0) {
                                        current_image_set_overlay.context2d().moveTo(viewer_point.x, viewer_point.y);
                                    }
                                    else {
                                        current_image_set_overlay.context2d().lineTo(viewer_point.x, viewer_point.y);
                                    }
                                }

                        
                                current_image_set_overlay.context2d().closePath();
                                current_image_set_overlay.context2d().stroke();
                                if (overlay_appearance["style"][key] == "fillRect") {
                                    current_image_set_overlay.context2d().fill();
                                }
                        
                            }
                        }
                        else {


                            let visible_inds = [];
                            for (let i = 0; i < boxes_to_add[key]["boxes"].length; i++) {

                                let box = boxes_to_add[key]["boxes"][i];


                                if (((box[1] < max_x) && (box[3] > min_x)) && ((box[0] < max_y) && (box[2] > min_y))) {
                                    visible_inds.push(i);
                                }
                            }
                            if (visible_inds.length <= MAX_BOXES_DISPLAYED) {
                                for (let ind of visible_inds) {

                                    let box = boxes_to_add[key]["boxes"][ind];
                                    let cls = boxes_to_add[key]["classes"][ind];
                                    current_image_set_overlay.context2d().strokeStyle = overlay_appearance["colors"][key][cls];
                                    current_image_set_overlay.context2d().fillStyle = overlay_appearance["colors"][key][cls] + "55";


                                    let viewer_point = current_image_set_viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[1], box[0]));
                                    let viewer_point_2 = current_image_set_viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[3], box[2]));
                                    
                                    current_image_set_overlay.context2d().strokeRect(
                                        viewer_point.x,
                                        viewer_point.y,
                                        (viewer_point_2.x - viewer_point.x),
                                        (viewer_point_2.y - viewer_point.y)
                                    );

                                    if (overlay_appearance["style"][key] == "fillRect") {
                                        current_image_set_overlay.context2d().fillRect(
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
                },
                clearBeforeRedraw: true
            });            

        
            for (let i = 0; i < switch_model_data["inspected_model_log"]["image_sets"].length; i++) {
        
                let image_set = switch_model_data["inspected_model_log"]["image_sets"][i];
        
                let entry = create_image_set_details_table(
                    image_set["username"],
                    image_set["farm_name"],
                    image_set["field_name"],
                    image_set["mission_date"]
                );
        
        
        
        
                $("#model_image_sets").append(`<tr>` +
        
                    `<td><div class="button-black button-black-hover" style="width: ${image_set_entry_width}; margin: 0px 1px;" ` +
                    `onclick="change_image_set('${i}')">` +
                    entry +
                    `</div></td>` +
                    `</tr>`);
            }
            init_model_viewer();
            change_image_set(0);
        }
    });
}


function init_model_viewer() {

    $("#model_viewer").empty();

    model_viewer = OpenSeadragon({
        id: "model_viewer",
        sequenceMode: true,
        prefixUrl: get_AC_PATH() + "/osd/images/",
        showNavigator: false,
        maxZoomLevel: 1000,
        zoomPerClick: 1,
        nextButton: "next_ims_button",
        previousButton: "prev_ims_button",
        showNavigationControl: false,
    });




    model_overlay = model_viewer.canvasOverlay({


        onOpen: function() {

            let region = model_image_set_regions[model_viewer.currentPage()];
            if (region != null) {

                let content_size = model_viewer.world.getItemAt(0).getContentSize();
                let image_w = content_size.x;
                let image_h = content_size.y;
                let hw_ratio = image_h / image_w;
                
                let bounds = get_bounding_box_for_polygon(region);


                let viewport_bounds = [
                    bounds[1] / image_w,
                    (bounds[0] / image_h) * hw_ratio,
                    (bounds[3] - bounds[1]) / image_w,
                    ((bounds[2] - bounds[0]) / image_h) * hw_ratio
                ];

                model_image_set_cur_bounds = new OpenSeadragon.Rect(
                    viewport_bounds[0],
                    viewport_bounds[1],
                    viewport_bounds[2],
                    viewport_bounds[3]
                );
            }
            else {
                model_image_set_cur_bounds = null;
            }

        },
        onRedraw: function() {
            let cur_tiles_url = model_viewer.source.tilesUrl;
            let basename_url = basename(cur_tiles_url);
            let model_image_set_image_name = basename_url.substring(0, basename_url.length-6);
            let region = model_image_set_regions[model_viewer.currentPage()];
        
            let boxes_to_add = {};
            boxes_to_add["annotation"] = {};
            boxes_to_add["annotation"]["boxes"] = model_image_set_annotations[model_image_set_image_name]["boxes"];
            boxes_to_add["annotation"]["classes"] = model_image_set_annotations[model_image_set_image_name]["classes"];
                
            let viewer_bounds = model_viewer.viewport.getBounds();

            let hw_ratio = model_overlay.imgHeight / model_overlay.imgWidth;
            let min_x = Math.floor(viewer_bounds.x * model_overlay.imgWidth);
            let min_y = Math.floor((viewer_bounds.y / hw_ratio) * model_overlay.imgHeight);
            let viewport_w = Math.ceil(viewer_bounds.width * model_overlay.imgWidth);
            let viewport_h = Math.ceil((viewer_bounds.height / hw_ratio) * model_overlay.imgHeight);
            let max_x = min_x + viewport_w;
            let max_y = min_y + viewport_h;

            let bounds = get_bounding_box_for_polygon(region);

            min_y = Math.max(min_y, bounds[0]);
            min_x = Math.max(min_x, bounds[1]);
            max_y = Math.min(max_y, bounds[2]);
            max_x = Math.min(max_x, bounds[3]);

            let draw_order = ["annotation"];

            for (let key of draw_order) { 

                model_overlay.context2d().lineWidth = 2;

                let visible_inds = [];
                for (let i = 0; i < boxes_to_add[key]["boxes"].length; i++) {

                    let box = boxes_to_add[key]["boxes"][i];

                    if (((box[1] < max_x) && (box[3] > min_x)) && ((box[0] < max_y) && (box[2] > min_y))) {
                        visible_inds.push(i);
                    }
                }
                if (visible_inds.length <= MAX_BOXES_DISPLAYED) {
                    for (let ind of visible_inds) {
                        let box = boxes_to_add[key]["boxes"][ind];
                        let cls = boxes_to_add[key]["classes"][ind];
                        model_overlay.context2d().strokeStyle = overlay_appearance["colors"][key][cls];
                        model_overlay.context2d().fillStyle = overlay_appearance["colors"][key][cls] + "55";


                        let viewer_point = model_viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[1], box[0]));
                        let viewer_point_2 = model_viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[3], box[2]));
                        
                        model_overlay.context2d().strokeRect(
                            viewer_point.x,
                            viewer_point.y,
                            (viewer_point_2.x - viewer_point.x),
                            (viewer_point_2.y - viewer_point.y)
                        );

                        if (overlay_appearance["style"][key] == "fillRect") {
                            model_overlay.context2d().fillRect(
                                viewer_point.x,
                                viewer_point.y,
                                (viewer_point_2.x - viewer_point.x),
                                (viewer_point_2.y - viewer_point.y)
                            );
                        }
                    }
                }
            }

            if (region != null) {

                let image_px_width = model_overlay.imgWidth;
                let image_px_height = model_overlay.imgHeight;
        
                let inner_poly = region;
                let outer_poly = [
                    [0-1e6, 0-1e6], 
                    [0-1e6, image_px_width+1e6], 
                    [image_px_height+1e6, image_px_width+1e6],
                    [image_px_height+1e6, 0-1e6]
                ];
        
                model_overlay.context2d().fillStyle = "#222621";
                model_overlay.context2d().beginPath();
        
                for (let poly of [outer_poly, inner_poly]) {
        
                    for (let i = 0; i < poly.length+1; i++) {
                        let pt = poly[(i)%poly.length];
                        let viewer_point = model_viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[1], pt[0]));
        
                        if (i == 0) {
                            model_overlay.context2d().moveTo(viewer_point.x, viewer_point.y);
                        }
                        else {
                            model_overlay.context2d().lineTo(viewer_point.x, viewer_point.y);
                        }
                    }
                    model_overlay.context2d().closePath();
        
                }
                model_overlay.context2d().mozFillRule = "evenodd";
                model_overlay.context2d().fill("evenodd");
            }


            if (model_image_set_cur_bounds != null) {

                if (region != null) {
        
                    model_viewer.world.getItemAt(0).setClip(
                        new OpenSeadragon.Rect(
                            bounds[1],
                            bounds[0],
                            (bounds[3] - bounds[1]),
                            (bounds[2] - bounds[0])
                        )
                    );
                }
        
        
                withFastOSDAnimation(model_viewer.viewport, function() {
                    model_viewer.viewport.fitBounds(model_image_set_cur_bounds);
                });
                model_image_set_cur_bounds = null;
            }

        },
        clearBeforeRedraw: true
    });
}
let model_viewer;
let model_overlay;
let model_image_set_annotations;
let model_image_set_regions;
let model_image_set_dzi_image_paths;
let model_image_set_cur_bounds;
function change_image_set(image_set_index) {
    let image_set = switch_model_data["inspected_model_log"]["image_sets"][parseInt(image_set_index)];
    $.post($(location).attr("href"),
    {
        action: "fetch_model_annotations",
        model_creator: switch_model_data["inspected_model_log"]["model_creator"],
        model_name: switch_model_data["inspected_model_log"]["model_name"],
        username: image_set["username"],
        farm_name: image_set["farm_name"],
        field_name: image_set["field_name"],
        mission_date: image_set["mission_date"]
    },
    function(response, status) {

        if (response.error) {
            show_modal_message("Error", response.message);
        }
        else {

            model_image_set_annotations = response.annotations;
            model_image_set_dzi_image_paths = [];
            model_image_set_regions = [];
            model_image_set_cur_bounds = null;
            for (let image_name of Object.keys(model_image_set_annotations)) {

                for (let region_key of ["fine_tuning_regions", "test_regions"]) {
                    for (let i = 0; i < model_image_set_annotations[image_name][region_key].length; i++) {
                        model_image_set_dzi_image_paths.push(
                            get_AC_PATH() + "/usr/data/" + image_set["username"] + "/image_sets/" +
                                            image_set["farm_name"] + "/" +
                                            image_set["field_name"] + "/" +
                                            image_set["mission_date"] + "/" +
                                            "dzi_images" + "/" +
                                            image_name + ".dzi"
                        );
                        model_image_set_regions.push(model_image_set_annotations[image_name][region_key][i]);
                    }
                }
            }

            model_viewer.tileSources = model_image_set_dzi_image_paths;
            model_viewer.goToPage(0);

        }
    });
}


function get_filtered_model_list() {

    let filtered_models = [];
    for (let model of switch_model_data["models"]) {

        let keep = true;
        for (let filter_option of switch_model_data["filter_options"]) {

            if ($("#" + filter_option + "_filter").val() == "-- All --" || $("#" + filter_option + "_filter").val() === model[filter_option]) {
                keep = true;
            }
            else {
                keep = false;
                break;
            }
        }
        if (keep) {
            filtered_models.push(model);
        }
    }

        let sort_combo_0_val = $("#sort_combo_0").val();
        let sort_combo_1_val = $("#sort_combo_1").val();
        filtered_models.sort(function(a, b) {
            return a[sort_combo_0_val].localeCompare(b[sort_combo_0_val], undefined, {numeric: true, sensitivity: 'base'}) || 
                   a[sort_combo_1_val].localeCompare(b[sort_combo_1_val], undefined, {numeric: true, sensitivity: 'base'});
        });

    return filtered_models;
}


function create_models_selection_table() {
    $("#models_table").empty();
    let filtered_models = get_filtered_model_list();
    for (let model of filtered_models) {
                    
        let model_name = model["model_name"];
        let model_creator = model["model_creator"];

        let model_details_table = create_model_details_table(model_creator, model_name);
        let button_id = model_creator + "." + model_name;
        let warn_icon;
        if (model["image_set_used_to_train_model"]) {
            warn_icon = 
            `<div style="width: 35px">` +
                `<div style="margin: 0 auto; width: 26px; height: 26px; color: yellow; border: 1px solid white; border-radius: 100%;">` +
                    `<i class="fa-solid fa-triangle-exclamation" style="font-size: 15px; margin-left: 5px; margin-top: 4px;"></i>` +
                `</div>` +
            `</div>`;
        }
        else {
            warn_icon = `<div style="width: 35px"></div>`;
        }
        $("#models_table").append(
            `<tr style="border-bottom: 1px solid white; border-color: #4c6645;">` + 
                `<td>` +
                    `<div class="table_entry" style="width: 330px; text-align: left;">${model_details_table}</div>` +
                `</td>` +
                `<td style="width: 100%">` +
                `</td>` +
                `<td>` +
                    `<button id="${button_id}" onclick="select_model('${model_creator}', '${model_name}')" style="font-size: 14px; width: 80px" class="button-green button-green-hover">Select</button>` + 
                `</td>` +
                `<td>` +
                    `<div style="width: 5px"></div>` +
                `</td>` +
                `<td>` +
                    `<button onclick="show_model_details('${model_creator}', '${model_name}')" style="font-size: 14px; width: 80px" class="button-green button-green-hover">Inspect</button>` +
                `</td>` +
                `<td>` +
                    `<div style="width: 5px"></div>` +
                `</td>` +
                `<td>` +
                    warn_icon +
                `</td>` +
                `<td>` +
                    `<div style="width: 5px"></div>` +
                `</td>` +
            `</tr>`);
    }
}

function select_model(model_creator, model_name) {
    let prev_model_creator = switch_model_data["selected_model"]["model_creator"];
    let prev_model_name = switch_model_data["selected_model"]["model_name"];

    if (prev_model_name !== null && prev_model_creator !== null) {
        let prev_button_id = prev_model_creator + "\\." + prev_model_name;
        enable_green_buttons([prev_button_id]);
    }

    switch_model_data["selected_model"] = {
        "model_creator": model_creator,
        "model_name": model_name

    }
    let button_id = model_creator + "\\." + model_name;
    disable_green_buttons([button_id]);
    enable_green_buttons(["submit_model_change"]);
}


function set_model_weights_to_random() {

    disable_model_actions();
    close_modal();

    $.post($(location).attr("href"),
    {
        action: "switch_model", 
        model_name: "Random Weights",
        model_creator: "",
    },
    function(response, status) {

        if (response.error) {
            show_modal_message("Error", response.message);
        }
    });
}


function show_models() {

    switch_model_data["models"] = [];
    switch_model_data["selected_model"] = {
        "model_creator": null,
        "model_name": null
    };

    $("#model_info").empty();
    $("#model_info").append(`<div class="loader"></div>`);



    $.post($(location).attr("href"),
    {
        action: "fetch_models",
        object_classes: (metadata["object_classes"]).join(",")
    },
    function(response, status) {
        if (response.error) {
            show_modal_message("Error", response.message);
        }
        else {
            $("#model_info").empty();

            switch_model_data["models"] = response.models;

            let models = switch_model_data["models"];
            
            if (models.length == 0) {
                $("#model_info").append(`<table><tr><td>No Models Found!</td></tr></table>`);
            }
            else {

                $("#model_info").append(
                    `<table>` +
                        `<tr>` +
                            `<td>` +
                                `<div style="width: 5px"></div>` +
                            `</td>` +
                            `<td>` +
                                `<table>` +
                                    `<tr>` +
                                        `<td>` +
                                            `<h class="header2" style="width: 150px; padding-left: 10px">Models</h>` +
                                        `</td>` +
                                    `</tr>` +
                                    `<tr>` +
                                        `<td>` +
                                            `<div class="scrollable_area" style="height: 350px; width: 800px; border: 1px solid white; overflow-y: scroll; border-radius: 10px;">` +
                                                `<table id="models_table" style="border-collapse: collapse;"></table>` +
                                            `</div>` +
                                        `</td>` +
                                    `</tr>` +
                                `</table>` +
                            `</td>` +
                            `<td style="width: 100%">` +
                            `</td>` +

                            `<td>` +
                                `<table>` + 
                                    `<tr>` +
                                        `<td>` +
                                            `<h class="header2" style="width: 150px; padding-left: 10px">Filter</h>` +
                                        `</td>` +
                                    `</tr>` +
                                    `<tr>` +
                                        `<td>` +
                                            `<div style="height: 100px; border: 1px solid white; border-radius: 10px; padding: 10px">` +
                                                `<table id="filter_table"></table>` +
                                            `</div>` +
                                        `</td>` +
                                    `</tr>` +
                                    `<tr>` +
                                        `<td>` +
                                            `<div style="height: 5px"></div>` +
                                        `</td>` +
                                    `</tr>` +
                                    `<tr>` +
                                        `<td>` +
                                            `<h class="header2" style="width: 150px; padding-left: 10px">Sort Order</h>` +
                                        `</td>` +
                                    `</tr>` +
                                    `<tr>` +
                                        `<td>` +
                                            `<div style="height: 100px; border: 1px solid white; border-radius: 10px; padding: 10px">` +
                                                `<table id="sort_table"></table>` +
                                            `</div>` +
                                        `</td>` +
                                    `</tr>` +
                                    `<tr>` +
                                        `<td>` +
                                            `<div style="height: 25px"></div>` +
                                        `</td>` +
                                    `</tr>` +                                    
                                    `<tr>` +
                                        `<td>` +
                                            `<div style="height: 75px; border: 1px solid white; border-radius: 10px; padding: 10px">` +
                                                `<div style="display: inline; color: yellow; border: 1px solid white; border-radius: 100%; padding: 3px 4px">` +
                                                    `<i class="fa-solid fa-triangle-exclamation" style="font-size: 15px"></i>` +
                                                `</div>` +
                                                `<div style="margin-left: 5px; line-height: 1.6; display: inline; font-size: 11px"> This symbol indicates that the model was trained with data from the current image set. Be aware that test data from this image set may have been used during model training.` +
                                                `</div>` +
                                            `</div>` +
                                        `</td>` +
                                    `</tr>` +

                                `</table>` +
                            `</td>` +
                            `<td>` +
                                `<div style="width: 5px"></div>` +
                            `</td>` +
                        `</tr>` +
                    `</table>`);
                    $("#model_info").append(
                    `<div style="text-align: center;">` +
                        `<div style="height: 30px"></div>` +
                        `<button id="submit_model_change" class="button-green button-green-hover" style="width: 240px">Switch To Selected Model</button>` +
                        `<div style="height: 10px"></div>` +
                        `</div>`);
                disable_green_buttons(["submit_model_change"]);


                let filter_values = {
                    "model_name": [],
                    "model_creator": []
                };
                let default_creator = "-- All --";
                for (let model of models) {
                    
                    let model_name = model["model_name"];
                    let model_creator = model["model_creator"];

                    filter_values["model_name"].push(model_name);
                    filter_values["model_creator"].push(model_creator);

                }

                let option_key_to_label = {
                    "model_creator": "Model Creator",
                    "model_name": "Model Name"
                }
                switch_model_data["filter_options"] = ["model_creator"];
                switch_model_data["sort_options"] = ["model_creator", "model_name"];

                for (let i = 0; i < switch_model_data["sort_options"].length; i++) {

                    
                    let select_id = "sort_combo_" + i;
                    $("#sort_table").append(
                        `<tr>` +
                        
                            `<td>` +
                                `<div style="width: 50px"></div>` +
                            `</td>` +
                            `<td>` +
                                `<div style="text-align: center; width: 200px">` +
                                    `<select id="${select_id}" class="nonfixed_dropdown" style="font-size: 14px"></select>` +
                                `</div>` +
                            `</td>` +
                            `<td>` +
                            `<div style="width: 50px"></div>` +
                        `</td>` +
                        `</tr>`
                    );
                    for (let j = i; j < switch_model_data["sort_options"].length; j++) {
                        $("#" + select_id).append($('<option>', {
                            value: switch_model_data["sort_options"][j],
                            text: option_key_to_label[switch_model_data["sort_options"][j]]
                        }));
                    }

                    if (i < switch_model_data["sort_options"].length - 1) {
                        $("#" + select_id).change(function() {
                            let select_num = parseInt(select_id[select_id.length-1]);
                            let next_id = "sort_combo_" + (select_num+1);
                            let selected_vals = [];
                            for (let k = 0; k <= select_num; k++) {
                                selected_vals.push($("#sort_combo_" + k).val());
                            }
                            $("#" + next_id).empty();
                            for (let sort_option of switch_model_data["sort_options"]) {
                                if (!(selected_vals.includes(sort_option))) {
                                    $("#" + next_id).append($('<option>', {
                                        value: sort_option,
                                        text: option_key_to_label[sort_option]
                                    }));
                                }
                            }

                            $("#" + next_id).val($("#" + next_id + ":first").val()).change();
                        });
                    }
                    else {
                        $("#" + select_id).change(function() {
                            create_models_selection_table();
                        });
                    }

                }

                for (let filter_option of switch_model_data["filter_options"]) {

                    let disp_text = option_key_to_label[filter_option];
                    let select_id = filter_option + "_filter";

                    $("#filter_table").append(
                        `<tr>` +
                            `<td>` +
                                `<div style="width: 100px; text-align: right; margin-right: 10px; font-size: 14px">${disp_text}</div>` +
                            `</td>` +
                            `<td>` +
                                `<div style="width: 180px">` +
                                    `<select id="${select_id}" class="nonfixed_dropdown" style="font-size: 14px"></select>` +
                                `</div>` +
                            `</td>` +
                        `</tr>`
                    );



                    $("#" + select_id).append($('<option>', {
                        value: "-- All --",
                        text: "-- All --"
                    }));
                    let unique_filter_values = natsort([... new Set(filter_values[filter_option])]);
                    for (let value of unique_filter_values) {
                        $("#" + select_id).append($('<option>', {
                            value: value,
                            text: value
                        }));
                    }

                    $("#" + select_id).change(function() {
                        create_models_selection_table();
                    });
                }

                $("#model_creator_filter").val(default_creator);

                create_models_selection_table();
            }

            $("#modal").css("display", "block");

            $("#submit_model_change").click(function() {

                
                let new_model_name = switch_model_data["selected_model"]["model_name"];
                let new_model_creator = switch_model_data["selected_model"]["model_creator"];
                switch_model(new_model_creator, new_model_name);

            });

        }
    });
}


function change_model() {
    show_modal_message(
        `Select Model`,
        `<div style="height: 500px">` +
            `<div style="height: 35px">` +
                `<table>` +
                    `<tr>` +
                        `<td>` +
                            `<button id="model_select_back_button" onclick="change_model()" class="button-green button-green-hover" style="margin-bottom: 10px; margin-left: 10px; width: 100px; font-size: 13px; border-radius: 5px; display: none">` +
                                    `<i class="fa-solid fa-caret-left" style="margin-right: 5px"></i>` +
                                    `Back` +
                            `</button>` +
                            `</td>` +
                        `<td style="width: 100%">` +

                        `</td>` +
                        `<td>` +
                            `<button id="random_weights_button" onclick="set_model_weights_to_random()" class="button-green button-green-hover" style="width: 145px; font-size: 13px; border-radius: 5px;">` +
                                `<i class="fa-solid fa-dice" style="margin-right: 5px"></i>` +
                                    `Set Model Weights To Random Values` +
                            `</button>` +
                        `</td>` +
                        `<td>` +
                            `<div style="width: 5px"></div>` +
                        `</td>` +
                    `</tr>` +
                `</table>` +
            `</div>` +
            `<div id="model_info">` +
                
            `</div>` +
        `</div>`

    , modal_width=1200, display=false);

    $("#model_select_back_button").hide();
    $("#random_weights_button").show();

    show_models();
}



function add_prediction_buttons() {
    let navigation_type = $('#navigation_dropdown').val();
    let single_text;
    let all_text;
    let multiple = Object.keys(annotations).length > 1;
    if (navigation_type === "images") {
        single_text = "Current Image";
        all_text = "All Images";
    }
    else {
        single_text = "Current Region";
        all_text = "All Regions";
        let num_regions = 0;
        for (let image_name of Object.keys(annotations)) {
            for (let region_key of ["regions_of_interest", "fine_tuning_regions", "test_regions"]) {
                num_regions += annotations[image_name][region_key].length;
            }
        }
        multiple = num_regions > 1;
    }


    $("#predict_single_button").addClass("no-transition");
    $("#predict_all_button").addClass("no-transition");

    if (multiple) {
        $("#predict_single_button").css({"width": "130px", "border-radius": "30px 0px 0px 30px"});
        $("#predict_single_button").text(single_text);
        $("#predict_all_button").css({"width": "130px", "border-radius": "0px 30px 30px 0px", "border-left": "none"});
        $("#predict_all_button").text(all_text);
        $("#predict_all_button").show();
    }
    else {
        $("#predict_single_button").css({"width": "260px", "border-radius": "30px 30px 30px 30px"});
        $("#predict_single_button").text(single_text);
        $("#predict_all_button").hide();
    }

    $("#predict_single_button")[0].offsetHeight;
    $("#predict_all_button")[0].offsetHeight;

    $("#predict_single_button").removeClass("no-transition");
    $("#predict_all_button").removeClass("no-transition");
}

function get_image_list_and_region_list_for_predicting_on_all(predict_on_images) {

    let image_list = [];
    let region_list = [];
    if (predict_on_images) {
        for (let image_name of Object.keys(annotations)) {
            let image_width = metadata["images"][image_name]["width_px"];
            let image_height = metadata["images"][image_name]["height_px"];
            image_list.push(image_name);
            region_list.push([
                [
                    [0, 0],
                    [0, image_width],
                    [image_height, image_width],
                    [image_height, 0]
                ]
            ]);
        }
    }
    else {
        for (let image_name of Object.keys(annotations)) {
            let image_region_list = [];
            for (let region_key of ["regions_of_interest", "fine_tuning_regions", "test_regions"]) {
                for (let region of annotations[image_name][region_key]) {
                    image_region_list.push(region);
                }
            }
            if (image_region_list.length > 0) {
                image_list.push(image_name);
                region_list.push(image_region_list);
            }
        }
    }

    return [image_list, region_list];

}



function switch_model(model_creator, model_name) {

    disable_model_actions();
    close_modal();

    $.post($(location).attr("href"),
    {
        action: "switch_model",
        model_name: model_name,
        model_creator: model_creator
    },
    function(response, status) {

        if (response.error) {
            show_modal_message("Error", response.message);
        }
    });
}


function create_navigator_viewer() {
    navigator_viewer = OpenSeadragon({
        id: "navigator_viewer",
        sequenceMode: true,
        prefixUrl: get_AC_PATH() + "/osd/images/",
        tileSources: dzi_image_paths,
        showNavigator: false,
        maxZoomLevel: 1000,
        zoomPerClick: 1,
        nextButton: "next-btn",
        previousButton: "prev-btn",
        showNavigationControl: false,
        imageSmoothingEnabled: true,
        zoomPerScroll: 1,
        panHorizontal: false,
        panVertical: false
    });
    
    navigator_overlay = navigator_viewer.canvasOverlay({
        clearBeforeRedraw: true
    });
}



function enable_model_actions() {
    let buttons = [
        "switch_model_button",
        "fine_tune_model_button",
        "predict_single_button", 
        "predict_all_button", 
        "request_result_button"
    ];
    enable_green_buttons(buttons);
}


function disable_model_actions() {
    let buttons = [
        "switch_model_button",
        "fine_tune_model_button",
        "predict_single_button", 
        "predict_all_button", 
        "request_result_button"
    ];
    disable_green_buttons(buttons);
}


$(document).ready(function() {

    image_set_info = data["image_set_info"];
    dzi_image_paths = data["dzi_image_paths"];
    annotations = data["annotations"];
    metadata = data["metadata"];
    camera_specs = data["camera_specs"];
    excess_green_record = data["excess_green_record"];
    predictions = data["predictions"];
    overlay_appearance = data["overlay_appearance"];
    hotkeys = data["hotkeys"];
    tags = data["tags"];


    if (data["maintenance_time"] !== "") {
        $("#maintenance_message").html("Site maintenance is scheduled for " + data["maintenance_time"] + ".");
        $("#maintenance_message").show();
    }

    set_heights();
    resize_window();

    initialize_class_select("class_select");
    initialize_class_select("map_builder_class_select", add_all_objects_option=true);
    initialize_class_select("pred_class_select", add_all_objects_option=true);

    add_prediction_buttons();

    create_overlays_table();

    create_navigator_viewer();

    update_overlay_color_css_rules();

    disable_green_buttons(["prev_image_button"]);
    if (dzi_image_paths.length == 1) {
        disable_green_buttons(["next_image_button"]);
    }

    $("#image_set_name").html(`<table><tr>` +
                                `<td>${image_set_info["farm_name"]}</td>` +
                                `<td style="width: 40px"></td>` +
                                `<td>${image_set_info["field_name"]}</td>` +
                                `<td style="width: 40px"></td>` +                                
                                `<td>${image_set_info["mission_date"]}</td>` +
                                `</tr></table>`
                                );

    image_to_dzi = {};
    for (let dzi_image_path of dzi_image_paths) {
        let image_name = basename(dzi_image_path);
        let extensionless_name = image_name.substring(0, image_name.length - 4);
        image_to_dzi[extensionless_name] = dzi_image_path;
    }

    let init_image_name = basename(dzi_image_paths[0]);
    cur_img_name = init_image_name.substring(0, init_image_name.length - 4);
    cur_region_index = -1;

    cur_view = "image";
    cur_panel = "annotation";


    if (metadata["is_ortho"]) {
        $("#apply_threshold_to_all_button").hide();
    }

    create_navigation_table();

    update_navigation_dropdown();


    if ((can_calculate_density(metadata, camera_specs))) {
        gsd = get_gsd();
    }

    
    let socket = io(
    "", {
        path: get_AC_PATH() + "/socket.io"
    });

    socket.emit("join_workspace", username + "/" + image_set_info["farm_name"] + "/" + image_set_info["field_name"] + "/" + image_set_info["mission_date"]);

    socket.on("workspace_occupied", function(update) {
        window.location.href = get_AC_PATH() + "/home/" + username;
    });


    socket.on("workers_update", function(update) {

        let num_workers = parseInt(update["num_workers"]);
        let num_workers_text;
        if (num_workers == 1) {
            num_workers_text = "worker thread available.";
        }
        else {
            num_workers_text = "worker threads available.";
        }

        $("#num_workers").html(num_workers);
        $("#num_workers_text").html(num_workers_text);

    });

    socket.on("image_set_update", function(update) {

        let state_name = update["state_name"];
        let progress = update["progress"];
        let error_message = update["error_message"];
        let prediction_image_names = update["prediction_image_names"];


        model_unassigned = update["model_name"] === "";
        if (model_unassigned) {
            $("#model_name").html("No model selected.");
        }
        else {
            let model_name = update["model_name"];
            $("#model_name").html(model_name);
            $("#image_set_state").html(state_name);
        }

        
        if (error_message === "") {
            $("#image_set_state_progress").html(progress);
        }
        else {
            $("#image_set_state_progress").html("-- ERROR --");
        }

        if (state_name === "Idle" && error_message === "") {
            enable_model_actions();
        }
        else {
            disable_model_actions();
        }





        if (error_message !== "") {
            let error_message = `An error has occurred: <br><br>` + update["error_message"] +
                                `<br><br>Please report this error to the site administrator.`;
            show_modal_message("Error", error_message);

        }
        if (prediction_image_names !== "") {
            
            $.post($(location).attr('href'),
            {
                action: "retrieve_predictions",
                image_names: prediction_image_names
            },
        
            function(response, status) {
        
                if (response.error) {
                    show_modal_message("Error", response.message);
        
                }
                else {

                    let prediction_image_name_lst = prediction_image_names.split(",");

                    for (let prediction_image_name of prediction_image_name_lst) {
                        predictions[prediction_image_name] = response.predictions[prediction_image_name];

                        if (prediction_image_name in voronoi_data && "prediction" in voronoi_data[prediction_image_name]) {
                            delete voronoi_data[prediction_image_name]["prediction"];
                        }
                    }

                    if ((cur_panel === "prediction") && (prediction_image_names.includes(cur_img_name))) {
                        show_prediction();
                    }
                }
            });
            
        }


    });


    update_count_combo(false);

    $("#chart_combo").change(function() {
        set_count_chart_data();
        update_count_chart();
    });

    $("#pred_class_select").change(function() {
        viewer.raiseEvent('update-viewport');

        set_count_chart_data();
        update_count_chart();

        create_overlays_table();
        
        set_score_chart_data();
        update_score_chart();

    })


    set_count_chart_data();
    set_score_chart_data();
    
    draw_count_chart();
    draw_score_chart();


    $("#tile_size_slider").change(function() {
        let slider_val = Number.parseFloat($("#tile_size_slider").val()).toFixed(2);
        $("#tile_size_slider_val").html(slider_val + " m");
    });

    $("#tile_size_slider").on("input", function() {
        let slider_val = Number.parseFloat($("#tile_size_slider").val()).toFixed(2);
        $("#tile_size_slider_val").html(slider_val + " m");
    });

    $("#tile_size_down").click(function() {
        lower_tile_size_slider();
    });

    $("#tile_size_up").click(function() {
        raise_tile_size_slider();
    });


    if (can_calculate_density(metadata, camera_specs)) {
        if (metadata["is_ortho"] || Object.keys(annotations).length >= 3) {


            $("#view_button_container").show();


            $("#view_button").click(function() {
                if (cur_view == "image") {
                    show_map();
                }
                else {
                    show_image(cur_img_name);
                }
            });
        }

        if (metadata["is_ortho"]) {
            let tile_size_range = calculate_tile_size_slider_range();
            $("#tile_size_slider").prop("min", tile_size_range[0]);
            $("#tile_size_slider").prop("max", tile_size_range[1]);
            $("#tile_size_slider").prop("value", tile_size_range[0]);
            $("#tile_size_slider_val").html(tile_size_range[0] + " m");
            $("#map_tile_size_controls").show();
        }
        else {
            $("#map_tile_size_controls").hide();
        }
    }
    
    show_image(cur_img_name);
    
    $("#save_button").click(async function() {
        await unselect_selected_annotation();
        save_annotations();
    });

    $("#next_tile_button").click(function() {
        grid_zoomed = false;
        if (cur_gridview_tile_index < cur_gridview_tiles.length-1) {
            cur_gridview_tile_index++;
        }
        viewer.raiseEvent('update-viewport');
        
        $("#gridview_info").text(
            `Current Tile: ${cur_gridview_tile_index+1} / ${cur_gridview_tiles.length}`
        );

        if (cur_gridview_tile_index == cur_gridview_tiles.length-1) {
            disable_green_buttons(["next_tile_button"]);
        }

        enable_green_buttons(["prev_tile_button"]);


    });


    $("#prev_tile_button").click(function() {
        grid_zoomed = false;

        if (cur_gridview_tile_index > 0) {
            cur_gridview_tile_index--;
        }
        viewer.raiseEvent('update-viewport');

        $("#gridview_info").text(
            `Current Tile: ${cur_gridview_tile_index+1} / ${cur_gridview_tiles.length}`
        );

        if (cur_gridview_tile_index == 0) {
            disable_green_buttons(["prev_tile_button"]);
        }
        enable_green_buttons(["next_tile_button"]);

    });

    $("#exit_gridview_button").click(function() {
        $("#control_panel").show();
        $("#gridview_panel").hide();

        viewer.zoomPerScroll = 1.2;
        viewer.panHorizontal = true;
        viewer.panVertical = true;
        overlay.onRedraw = anno_and_pred_onRedraw;
        set_cur_bounds();

        let dzi_image_path = image_to_dzi[cur_img_name];
        viewer.open(dzi_image_path);
    })

    $("#gridview_button").click(function() {

        $("#control_panel").hide();
        $("#gridview_panel").show();

        $("#engage_grid").show();
        $("#disengage_grid").hide();
        $("#engaged_grid_controls").hide();

    
        $("#grid_overlap_percent_input").prop("disabled", false);
        $("#grid_setting_controls").css("opacity", 1.0);

    });

    $("#disengage_grid").click(function() {
        $("#engage_grid").show();
        $("#disengage_grid").hide();
        $("#engaged_grid_controls").hide();

        $("#grid_overlap_percent_input").prop("disabled", false);
        $("#grid_setting_controls").css("opacity", 1.0);

        viewer.zoomPerScroll = 1.2;
        viewer.panHorizontal = true;
        viewer.panVertical = true;
        overlay.onRedraw = anno_and_pred_onRedraw;

        let image_width_px = metadata["images"][cur_img_name]["width_px"];
        let image_height_px = metadata["images"][cur_img_name]["height_px"];


        let cur_tile = cur_gridview_tiles[cur_gridview_tile_index];
        let hw_ratio = image_height_px / image_width_px;
        
        let viewport_bounds = [
            cur_tile[1] / image_width_px,
            (cur_tile[0] / image_height_px) * hw_ratio,
            (cur_tile[3] - cur_tile[1]) / image_width_px,
            ((cur_tile[2] - cur_tile[0]) / image_height_px) * hw_ratio
        ];

        cur_bounds = new OpenSeadragon.Rect(
            viewport_bounds[0],
            viewport_bounds[1],
            viewport_bounds[2],
            viewport_bounds[3]
        );


        let dzi_image_path = image_to_dzi[cur_img_name];
        viewer.open(dzi_image_path);

    });


    $("#grid_overlap_percent_input").on("input", function(e) {
        if (update_grid_overlap_percent()) {
            enable_green_buttons(["engage_grid"]);
        }
        else {
            disable_green_buttons(["engage_grid"]);
        }
    });


    $("#engage_grid").click(function() {

        $("#engaged_grid_controls").show();


        $("#engage_grid").hide();
        $("#disengage_grid").show();

        $("#grid_overlap_percent_input").prop("disabled", true);
        $("#grid_setting_controls").css("opacity", 0.5);

        let viewer_bounds = viewer.viewport.getBounds();
    
        let hw_ratio = overlay.imgHeight / overlay.imgWidth;
        let min_x = Math.floor(viewer_bounds.x * overlay.imgWidth);
        let min_y = Math.floor((viewer_bounds.y / hw_ratio) * overlay.imgHeight);
        let viewport_w = Math.ceil(viewer_bounds.width * overlay.imgWidth);
        let viewport_h = Math.ceil((viewer_bounds.height / hw_ratio) * overlay.imgHeight);
        let max_x = min_x + viewport_w;
        let max_y = min_y + viewport_h;

        let tile_w = max_x - min_x;
        let tile_h = max_y - min_y;

        let tile_size = Math.min(tile_w, tile_h);

        let region_min_y;
        let region_min_x;
        let region_max_y;
        let region_max_x;

        let navigation_type = $("#navigation_dropdown").val()
        if (navigation_type === "images") {
            region_min_y = 0;
            region_min_x = 0
            region_max_y = metadata["images"][cur_img_name]["height_px"];
            region_max_x = metadata["images"][cur_img_name]["width_px"];
        }
        else {
            let region = annotations[cur_img_name][navigation_type][cur_region_index];
            let region_bbox = get_bounding_box_for_polygon(region);
            region_min_y = region_bbox[0];
            region_min_x = region_bbox[1];
            region_max_y = region_bbox[2];
            region_max_x = region_bbox[3];
        }

        let tile_overlap_percent = parseFloat($("#grid_overlap_percent_input").val());
        let overlap_px = Math.floor(tile_size * tile_overlap_percent);

        let subject_polygon = [];
        if (navigation_type !== "images") {
            let region = annotations[cur_img_name][navigation_type][cur_region_index];
            for (let c of region) {
                subject_polygon.push([c[1], c[0]]);
            }
        }

        let gridview_tiles = []
        let col_covered = false;
        let tile_min_y = region_min_y;
        while (!col_covered) {
            let tile_max_y = tile_min_y + tile_size;
            if (tile_max_y >= region_max_y) {
                tile_max_y = region_max_y;
                tile_min_y = tile_max_y - tile_size;
                col_covered = true;
            }

            let row_covered = false;
            let tile_min_x = region_min_x;
            while (!row_covered) {
                let tile_max_x = tile_min_x + tile_size;
                if (tile_max_x >= region_max_x) {
                    tile_max_x = region_max_x;
                    tile_min_x = tile_max_x - tile_size;
                    row_covered = true;
                }

                let tile = [tile_min_y, tile_min_x, tile_max_y, tile_max_x];

                let add = true;
                if (navigation_type !== "images") {

                    let clip_polygon = [
                        [tile_min_x, tile_min_y],
                        [tile_max_x, tile_min_y],
                        [tile_max_x, tile_max_y],
                        [tile_min_x, tile_max_y]
                    ];

                    let clipped_polygon = clip_polygons_xy(subject_polygon, clip_polygon);
                    if (get_polygon_area(clipped_polygon) == 0) {
                        add = false;
                    }
                }
                
                if (add) {
                    gridview_tiles.push(tile);
                }
                tile_min_x += (tile_size - overlap_px);
            }
            tile_min_y += (tile_size - overlap_px);    
        }

        cur_gridview_tiles = gridview_tiles;
        cur_gridview_tile_index = 0;


        viewer.zoomPerScroll = 1;
        viewer.panHorizontal = false;
        viewer.panVertical = false;

        overlay.onOpen = function() {

            $("#gridview_info").text(
                `Current Tile: ${cur_gridview_tile_index+1} / ${cur_gridview_tiles.length}`
            );

            if (cur_gridview_tile_index == 0) {
                disable_green_buttons(["prev_tile_button"]);
            }
            else {
                enable_green_buttons(["prev_tile_button"]);
            }

            if (cur_gridview_tile_index == cur_gridview_tiles.length - 1) {
                disable_green_buttons(["next_tile_button"]);
            }
            else {
                enable_green_buttons(["next_tile_button"]);
            }

            grid_zoomed = false;
            navigator_viewer.open(dzi_image_path);
        }

        overlay.onRedraw = gridview_onRedraw;

        let dzi_image_path = image_to_dzi[cur_img_name];
        viewer.open(dzi_image_path);



    });

    $("#help_button").click(function() {

        let head = "Help";
        let create_annotation_text = (hotkeys["Create Annotation"]).toUpperCase();
        if (create_annotation_text === " ") {
            create_annotation_text = "SPACE";
        }
        let delete_annotation_text = (hotkeys["Delete Annotation"]).toUpperCase();
        let message = `<div style="line-height: 150%; padding: 10px">&#8226; Hold the <span style='border: 1px solid white; font-size: 14px; padding: 5px 10px; margin: 0px 5px'>` + create_annotation_text + `</span> key and left mouse button to create a new box annotation or region.` +
        `<br><br>&#8226; When creating a region, double click the left mouse button to complete the region.` + 
        `<br><br>&#8226; Click on an existing box annotation / region to select it and change its boundaries.` +
        `<br><br>&#8226; Use the <span style='border: 1px solid white; font-size: 14px; padding: 5px 10px; margin: 0px 5px'>` + delete_annotation_text + `</span> key to remove whichever box annotation / region is currently selected.` + 
        // `<br><br>&#8226; If a test region is selected, pressing the <span style='border: 1px solid white; font-size: 16px; padding: 5px 10px; margin: 0px 5px'>m</span> key will change that region into a fine-tuning region.` + 
        // `<br><br>&#8226; Use the number keys to switch between different object classes.` + 
        `<br><br>&#8226; Don't forget to save your work!</div>`;
        show_modal_message(head, message, modal_width=750);
    });

    $("#request_result_button").click(function() {
        if (model_unassigned) {
            show_modal_message("No Model Selected", "A model must be selected before predictions can be generated.");
        }
        else {
            submit_result_request();
        }
    });


    $("#fine_tune_model_button").click(function() {
        if (model_unassigned) {
            show_modal_message("No Model Selected", "A model must be selected before fine-tuning can be applied.");
        }
        else {
            show_fine_tuning_modal();
        }
    });

    $("#switch_model_button").click(function() {
        change_model();
    });


    $("#use_predictions_button").click(function() {


        let navigation_type = $('#navigation_dropdown').val();
        let roi;
        if (navigation_type === "images") {
            roi = "image";
        }
        else {
            roi = "region";
        }

        let cur_class_ind = parseInt($("#pred_class_select").val());
        let object_str;
        if (cur_class_ind == -1) {
            object_str = "";
        }
        else {
            object_str = (metadata["object_classes"][cur_class_ind]).toLowerCase();
        }


        show_modal_message(
            `Are you sure?`,
            `<div>This action will remove all existing ${object_str} annotations for this ${roi}.</div>` +
            `<div style="height: 10px"></div>` +
            `<table>` +
                `<tr>` +
                    `<td style="width: 50%"></td>` +
                    `<td>` +
                        `<button class="button-green button-green-hover" `+
                            `style="width: 150px" onclick="confirmed_use_predictions()">Continue` +
                        `</button>` +
                    `</td>` +
                    `<td>` +
                        `<div style="width: 10px"></div>` +
                    `</td>` +
                    `<td>` +
                        `<button class="button-green button-green-hover" ` +
                            `style="width: 150px" onclick="close_modal()">Cancel` +
                        `</button>` +
                    `</td>` +
                    `<td style="width: 50%"></td>` +
                `</tr>` +
            `</table>`
        );
    });

    
    $("#overlays_table").change(function() {
        viewer.raiseEvent('update-viewport');
    });


    $("#scores_switch").change(function() {
        viewer.raiseEvent('update-viewport');
    });



    $("#confidence_slider").change(function() {
        let slider_val = Number.parseFloat($("#confidence_slider").val()).toFixed(2);
        $("#confidence_slider_val").html("> " + slider_val);

        if (cur_img_name in voronoi_data && "prediction" in voronoi_data[cur_img_name]) {
            delete voronoi_data[cur_img_name]["prediction"];
        }

        viewer.raiseEvent('update-viewport');
        set_count_chart_data();
        update_count_chart();
        update_score_chart();
    });

    $("#confidence_slider").on("input", function() {
        let slider_val = Number.parseFloat($("#confidence_slider").val()).toFixed(2);
        $("#confidence_slider_val").html("> " + slider_val);
    });

    $("#score_down").click(function() {
        lower_slider();
    });

    $("#score_up").click(function() {
        raise_slider();
    });

    $("#apply_threshold_to_all_button").click(function() {
        disable_green_buttons(["apply_threshold_to_all_button"]);
        let cur_val = parseFloat(parseFloat($("#threshold_slider").val()).toFixed(2));

        for (let image_name of Object.keys(excess_green_record)) {
            let prev_val = excess_green_record[image_name]; 
            excess_green_record[image_name] = cur_val;

            if (prev_val != cur_val) {
                $("#save_button").removeClass("button-green");
                $("#save_button").removeClass("button-green-hover");
                $("#save_button").addClass("button-red");
                $("#save_button").addClass("button-red-hover");
            }
        }
    });


    
    $("#next_image_button").click(function() {
        change_to_next_image();
    });

    $("#prev_image_button").click(function() {
        change_to_prev_image();
    });


    $("#enable_pan_button").click(function() {
        $("#panning_enabled_status").html("Yes");
        disable_green_buttons(["enable_pan_button"]);
        pan_viewport();
    });
    


    $("#segment_button").click(function() {
        disable_green_buttons(["segment_button"]);

        if ($("#segmentation_loader").hasClass('load-complete')) {
            $("#segmentation_loader").toggleClass('load-complete');
            $("#segmentation_checkmark").toggle();
        }
        $("#segmentation_loader").show();
        enable_green_buttons(["enable_pan_button"]);
        $("#panning_enabled_status").html("No");
        segment_viewport();

    });

    $("#class_select").change(async function() {
        $("#annotation_radio_input").prop("checked", true).change();
    });

    $("input[name=edit_layer_radio]").change(async function(e) {

        await unselect_selected_annotation();

        cur_edit_layer = $('input[name=edit_layer_radio]:checked').val();

        update_overlay_color_css_rules();



        if (cur_edit_layer === "annotation") {
            anno.setDrawingTool("rect");
        }
        else {
            anno.setDrawingTool("polygon");
        }

        if (cur_panel === "annotation") {
            overlay.clear();
            if ($("#engaged_grid_controls").is(":visible")) {
                gridview_onRedraw();
            }
            else {
                anno_and_pred_onRedraw();
            }
        }
    });


    $("#navigation_dropdown").change(function() {
        create_navigation_table();
        update_count_combo(false);
        add_prediction_buttons();

        let navigation_type = $("#navigation_dropdown").val();
        let region_radio_labels = ["region_of_interest_label", "fine_tuning_region_label", "test_region_label"];
        if ((navigation_type == "regions_of_interest") || (navigation_type === "fine_tuning_regions" || navigation_type == "test_regions")) {
            $("input:radio[name=edit_layer_radio]").prop("disabled", true);
            for (let radio_label of region_radio_labels) {
                $("#" + radio_label).css("opacity", 0.5);
                $("#" + radio_label).css("cursor", "default");
            }
            $("#show_segmentation_button").hide();
        }
        else {
            $("input:radio[name=edit_layer_radio]").prop("disabled", false);
            for (let radio_label of region_radio_labels) {
                $("#" + radio_label).css("opacity", 1.0);
                $("#" + radio_label).css("cursor", "pointer");
            }
            $("#show_segmentation_button").show();

        }
        if (cur_panel === "segmentation") {
            viewer = null;
            cur_panel = "annotation";
        }
        let disp_nav_item = null;
        for (let nav_item of cur_nav_list) {
            let image_name = nav_item.split("/")[0];
            if (image_name === cur_img_name) {
                disp_nav_item = nav_item;
                break;
            }
        }
        if (disp_nav_item == null) {
            change_image(cur_nav_list[0]);
        }
        else {
            change_image(disp_nav_item);
        }

        $("input:radio[name=edit_layer_radio]").filter("[value=annotation]").prop("checked", true).change();

    });








    $("#threshold_slider").on("input", function() {
        let cur_val = parseFloat($("#threshold_slider").val());
        $("#threshold_slider_val").html(cur_val.toFixed(2));
    });

    $("#threshold_slider").change(function() {
        $("#save_button").removeClass("button-green");
        $("#save_button").removeClass("button-green-hover");
        $("#save_button").addClass("button-red");
        $("#save_button").addClass("button-red-hover");
        let cur_val = parseFloat($("#threshold_slider").val());
        excess_green_record[cur_img_name] = cur_val;
        $("#threshold_slider_val").html(cur_val.toFixed(2));
        update_apply_current_threshold_to_all_images_button();
    });




    function raise_threshold_slider() {
        let slider_val = parseFloat($("#threshold_slider").val());
        if (slider_val < 2) {
            slider_val = slider_val + 0.01;
            $("#threshold_slider").val(slider_val).change();
        }
        update_apply_current_threshold_to_all_images_button();
    }
    function lower_threshold_slider() {
        let slider_val = parseFloat($("#threshold_slider").val());
        if (slider_val > -2) {
            slider_val = slider_val - 0.01;
            $("#threshold_slider").val(slider_val).change();
        }
        update_apply_current_threshold_to_all_images_button();
    }

    $("#threshold_score_down").click(function() {
        lower_threshold_slider();
    });

    $("#threshold_score_up").click(function() {
        raise_threshold_slider();
    });

    $("#upload_annotations_button").click(function() {

        show_modal_message(`Upload Annotations`, 
            `<div>Annotations must be provided as a single JSON file. The file must follow the format below.</div>` +
            `<div style="height: 10px"></div>` +
            `<table>` +
                `<tr>` +
                    `<td>` +
                        `<div style="text-align: center; width: 350px;">` +
                            `<textarea class="json_text_area" style="width: 300px; margin: 0 auto; height: 270px">${annotations_format_sample_text}</textarea>` +
                        `</div>` +
                    `</td>` +
                    `<td>` +

                        `<ul>` +

                        `<li>All boxes must be encoded with four values (in <span style="font-weight: bold">pixel coordinates</span>):` +
                        `<br>` +
                        `<span style="margin-left: 75px; font-family: 'Lucida Console', Monaco, monospace;">[ x_min, y_min, x_max, y_max ]</span> ` + 
                        `</li>` +
                        `<br>` +
                        `<br>` +
                        `<li>All regions must be encoded as lists of x, y coordinate pairs.` +
                        `</li>` +
                        `<br>` +
                        `<br>` +
                        `<li>Class indices begin at zero. Class ordering should match the ordering used elsewhere in the workspace page.` +
                        `</li>` +
                        `<br>` +
                        `<br>` +
                        `<li>Images that are present in the image set but not found in the uploaded file will be unaffected by the upload process.` +
                        `</li>` +
                        `<br>` +
                        `<br>` +
                        `</li>` +

                        `</ul>` +

                    `</td>` +
                    `<td>` +
                        `<div style="width: 10px"></div>` +
                    `</td>` +
                `</tr>` +
            `</table>` +

            `<div style="height: 10px"></div>` +
            
            `<form id="annotations_upload_form" action="">` +
            `<table>` + 
                `<tr>` + 
                    `<td>` +
                        `<div style="text-align: center">` +
                            `<div style="border: 1px solid white; border-radius: 8px; width: 425px; margin: 0 auto">` +
                                `<div id="annotations_dropzone" class="dropzone" style="height: 195px">` +
                                    `<div class="dz-message data-dz-message">` +
                                        `<span>Drop Annotations File Here</span>` +
                                    `</div>` +
                                `</div>`+
                            `</div>` +
                        `</div>` +
                    `</td>` +
                `</tr>` +
            `</table>` +
            `</form>` +
            `<div style="height: 10px"></div>` +
            `<div style="text-align: center">` +
                `<button style="width: 250px;" class="button-green button-green-hover" id="submit_annotations_button">Upload Annotations</button>` +
            `</div>`, 950
        );

        disable_green_buttons(["submit_annotations_button"]);
        let annotations_dropzone = new Dropzone("#annotations_dropzone", { 
            url: $(location).attr('href') + "/annotations_upload",
            autoProcessQueue: false,
            paramName: function(n) { return 'source_file[]'; },
            uploadMultiple: false,
            maxFilesize: 100
        });


        annotations_dropzone.on("success", function(file, response) {   

            annotations_dropzone.removeFile(file);
            if (annotations_dropzone.getAcceptedFiles().length == 0) {

                annotations_dropzone.removeAllFiles(true);
                annotations_dropzone.options.autoProcessQueue = false;
                annotations = response.annotations;

                update_navigation_dropdown();
                $("#navigation_dropdown").val("images").change();
                show_modal_message(`Success!`, `The annotation file you uploaded has been successfully processed.`);

            }
        });

        annotations_dropzone.on("error", function(file, response) {

            annotations_dropzone.options.autoProcessQueue = false;

            if (typeof(response) == "object" && "error" in response) {
                upload_error = response.error;
            }
            else {
                upload_error = response;
            }

            show_modal_message(`Error`, `An error occured while processing the uploaded annotations:<br><br>` + upload_error);

        });




        annotations_dropzone.on("addedfile", function() {
            enable_green_buttons(["submit_annotations_button"]);
        });


        $("#submit_annotations_button").click(function(e) {

            e.preventDefault();
            e.stopPropagation();

            annotations_dropzone.options.autoProcessQueue = true;
            annotations_dropzone.processQueue();
        });
    });


    $("#download_annotations_button").click(function() {

        show_modal_message(`Download Annotations`, 
            `<div>Annotations will be downloaded as a single JSON file of the following format:</div>` +
            `<div style="height: 10px"></div>` +

            `<table>` +
                `<tr>` +
                    `<td>` +
                        `<div style="text-align: center; width: 300px;">` +
                            `<textarea class="json_text_area" style="width: 300px; margin: 0 auto; height: 270px">${annotations_format_sample_text}</textarea>` +
                        `</div>` +
                    `</td>` +
                    `<td>` +

                        `<ul>` +

                        `<li>All boxes are encoded with four values (in <span style="font-weight: bold">pixel coordinates</span>):` +
                        `<br>` +
                        `<br>` +
                        `<div style="text-align: center">` +
                            `<div style="font-family: 'Lucida Console', Monaco, monospace;">[ x_min, y_min, x_max, y_max ]</div> ` + 
                        `</div>` +
                        `</li>` +
                        `<br>` +
                        `<br>` +
                        `<li>All regions are encoded as lists of x, y coordinate pairs.` +
                        `</li>` +
                        `<br>` +
                        `<br>` +
                        `<li>Class indices begin at zero. Class ordering matches the ordering used elsewhere in the workspace page.` +
                        `</li>` +
                        `</ul>` +
                    `</td>` +
                `</tr>` +
            `</table>` +
            `<div style="height: 20px"></div>` +
            `<div style="text-align: center">` +
                `<button style="width: 250px;" class="button-green button-green-hover" onclick="download_annotations()" id="prepare_download_button">Prepare Download</button>` +
            `</div>`
            , 950
        );

    });


    $("body").keydown(function(e) {
        let focus_els = $(":focus");
        if (focus_els.length > 0 && focus_els[0].id.startsWith("hotkey_")) {
            hotkey_change(focus_els[0].id, e);
        }
        else if (selected_annotation !== null) {
            selected_keydown_handler(e);
        }
        else if ($("#engaged_grid_controls").is(":visible")) {
            grid_keydown_handler(e);
        }
        else {
            keydown_handler(e);
        }
    });


    $("#image_visible_switch").change(function() {
        viewer.raiseEvent('update-viewport');
    });

    $("#predict_single_button").click(function() {
        if (model_unassigned) {
            show_modal_message("No Model Selected", "A model must be selected before predictions can be generated.");
        }
        else {
            let predict_single_callback = function() {

                let navigation_type = $('#navigation_dropdown').val();
                let image_list;
                let region_list;
                if (navigation_type === "images") {
                    let image_width = metadata["images"][cur_img_name]["width_px"];
                    let image_height = metadata["images"][cur_img_name]["height_px"];
                    image_list = [cur_img_name];
                    region_list = [[
                        [
                            [0, 0],
                            [0, image_width],
                            [image_height, image_width],
                            [image_height, 0]
                        ]
                    ]];
                }
                else {
                    image_list = [cur_img_name];
                    let region = annotations[cur_img_name][navigation_type][cur_region_index];
                    region_list = [[region]];
                }
    
                submit_prediction_request_confirmed(image_list, region_list, false, false, false);

            }
            
            save_annotations(predict_single_callback);


        }
    });

    $("#predict_all_button").click(function() {
        if (model_unassigned) {
            show_modal_message("No Model Selected", "A model must be selected before predictions can be generated.");
        }
        else {
            let predict_all_callback = function() {
                let navigation_type = $('#navigation_dropdown').val();
                let predict_on_images = navigation_type === "images";
                let res = get_image_list_and_region_list_for_predicting_on_all(predict_on_images);
                let image_list = res[0];
                let region_list = res[1];
                
                submit_prediction_request_confirmed(image_list, region_list, false, false, false);
            }

            save_annotations(predict_all_callback);


        }
    });

});



function download_annotations() {

    show_modal_message("Preparing Download", 
        `<div style="height: 50px">` +
            `<div id="prep_download_message">Preparing spreadsheet...</div>` +
            `<div id="prep_download_loader" class="loader"></div>` +
            `<div style="text-align: center; margin-top: 20px"><a class="button-black button-black-hover" id="download_button" style="padding: 10px; border-radius: 30px" download="annotations.json" hidden>` +
                `<i class="fa-solid fa-file-arrow-down"></i><span style="margin-left: 10px">Download Annotations</span></a>` +
            `</div>` +
        `</div>`);


    $.post($(location).attr('href'),
    {
        action: "download_annotations",
    },
    
    function(response, status) {

        $("#prep_download_loader").hide();

        if (response.error) {
            $("#modal_head_text").html("Error");
            $("#prep_download_message").html("An error occurred while generating the annotations file: " + response.message);
        }
        else {
            
            let download_path = get_AC_PATH() + "/usr/data/" + username + "/image_sets/" + image_set_info["farm_name"] + "/" + 
            image_set_info["field_name"] + "/" + image_set_info["mission_date"] + "/annotations/download_annotations.json";

            $("#download_button").attr("href", download_path);
            $("#modal_head_text").html("Ready For Download");
            $("#prep_download_message").html("The annotations file is ready for download.");
            $("#download_button").show();
        }
    });


}

function update_apply_current_threshold_to_all_images_button() {
    if (excess_green_values_are_all_the_same()) {
        disable_green_buttons(["apply_threshold_to_all_button"]);
    }
    else {
        enable_green_buttons(["apply_threshold_to_all_button"]);
    }
}

function excess_green_values_are_all_the_same() {
    let image_names = Object.keys(excess_green_record);
    for (let i = 1; i < image_names.length; i++) {
        if (excess_green_record[image_names[i]] != excess_green_record[image_names[i-1]]) {
            return false;
        }
    }
    return true;
}



function gridview_onRedraw() {

    let navigation_type = $("#navigation_dropdown").val();

    let boxes_to_add = {};

    boxes_to_add["region_of_interest"] = {};
    boxes_to_add["region_of_interest"]["boxes"] = annotations[cur_img_name]["regions_of_interest"];
    boxes_to_add["fine_tuning_region"] = {};
    boxes_to_add["fine_tuning_region"]["boxes"] = annotations[cur_img_name]["fine_tuning_regions"];
    boxes_to_add["test_region"] = {};
    boxes_to_add["test_region"]["boxes"] = annotations[cur_img_name]["test_regions"];
    boxes_to_add["annotation"] = {};
    boxes_to_add["annotation"]["boxes"] = annotations[cur_img_name]["boxes"];
    boxes_to_add["annotation"]["classes"] = annotations[cur_img_name]["classes"];

        
    let viewer_bounds = viewer.viewport.getBounds();
    let container_size = viewer.viewport.getContainerSize();

    let hw_ratio = overlay.imgHeight / overlay.imgWidth;
    let min_x = Math.floor(viewer_bounds.x * overlay.imgWidth);
    let min_y = Math.floor((viewer_bounds.y / hw_ratio) * overlay.imgHeight);
    let viewport_w = Math.ceil(viewer_bounds.width * overlay.imgWidth);
    let viewport_h = Math.ceil((viewer_bounds.height / hw_ratio) * overlay.imgHeight);
    let max_x = min_x + viewport_w;
    let max_y = min_y + viewport_h;

    let image_w_px = metadata["images"][cur_img_name]["width_px"];
    let image_h_px = metadata["images"][cur_img_name]["height_px"];



    let cur_tile = cur_gridview_tiles[cur_gridview_tile_index];



    overlay.context2d().font = "14px arial";


    let draw_order = overlay_appearance["draw_order"];
    for (let key of draw_order) {

        if (!(key in boxes_to_add)) {
            continue;
        }
        

        overlay.context2d().lineWidth = 2;


        if ((key === "region_of_interest" || key === "fine_tuning_region") || key === "test_region") {

            overlay.context2d().strokeStyle = overlay_appearance["colors"][key];
            overlay.context2d().fillStyle = overlay_appearance["colors"][key] + "55";


            for (let i = 0; i < boxes_to_add[key]["boxes"].length; i++) {

                if ((cur_edit_layer === key) && (i == selected_annotation_index)) {
                    continue;
                }
                let region = boxes_to_add[key]["boxes"][i];

                overlay.context2d().beginPath();
                for (let j = 0; j < region.length; j++) {
                    let pt = region[j];
        
                    let viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[1], pt[0]));
                    
                    if (j == 0) {
                        overlay.context2d().moveTo(viewer_point.x, viewer_point.y);
                    }
                    else {
                        overlay.context2d().lineTo(viewer_point.x, viewer_point.y);
                    }
                }
        
                overlay.context2d().closePath();
                overlay.context2d().stroke();
                if (overlay_appearance["style"][key] == "fillRect") {
                    overlay.context2d().fill();
                }
        
            }
        
        }
        else {

            let visible_inds = [];
            loop1:
            for (let i = 0; i < boxes_to_add[key]["boxes"].length; i++) {
                if ((cur_edit_layer === key) && (i == selected_annotation_index)) {
                    continue;
                }

                let box = boxes_to_add[key]["boxes"][i];

                if (((box[1] < max_x) && (box[3] > min_x)) && ((box[0] < max_y) && (box[2] > min_y))) {

                    visible_inds.push(i);
                    if (visible_inds.length > MAX_BOXES_DISPLAYED) {
                        break loop1;
                    }
                }

            }

            if (visible_inds.length <= MAX_BOXES_DISPLAYED) {
                for (let ind of visible_inds) {
                    let box = boxes_to_add[key]["boxes"][ind];
                    let cls = boxes_to_add[key]["classes"][ind];
                    overlay.context2d().strokeStyle = overlay_appearance["colors"][key][cls];
                    overlay.context2d().fillStyle = overlay_appearance["colors"][key][cls] + "55";



                    let viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[1], box[0]));
                    let viewer_point_2 = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(box[3], box[2]));

                    overlay.context2d().strokeRect(
                        viewer_point.x,
                        viewer_point.y,
                        (viewer_point_2.x - viewer_point.x),
                        (viewer_point_2.y - viewer_point.y)
                    );

                    if (overlay_appearance["style"][key] == "fillRect") {
                        overlay.context2d().fillRect(
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




    delete boxes_to_add["annotation"];
    navigator_overlay.clear(); 

    draw_order = overlay_appearance["draw_order"];
    for (let key of draw_order) {

        if (!(key in boxes_to_add)) {
            continue;
        }
        
        navigator_overlay.context2d().strokeStyle = overlay_appearance["colors"][key];
        navigator_overlay.context2d().fillStyle = overlay_appearance["colors"][key] + "55";
        navigator_overlay.context2d().lineWidth = 2;


        for (let i = 0; i < boxes_to_add[key]["boxes"].length; i++) {

            if ((cur_edit_layer === key) && (i == selected_annotation_index)) {
                continue;
            }
            let region = boxes_to_add[key]["boxes"][i];

            navigator_overlay.context2d().beginPath();
            for (let j = 0; j < region.length; j++) {
                let pt = region[j];
    
                let viewer_point = navigator_viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[1], pt[0]));
                
                if (j == 0) {
                    navigator_overlay.context2d().moveTo(viewer_point.x, viewer_point.y);
                }
                else {
                    navigator_overlay.context2d().lineTo(viewer_point.x, viewer_point.y);
                }
            }
    
            navigator_overlay.context2d().closePath();
            navigator_overlay.context2d().stroke();
            if (overlay_appearance["style"][key] == "fillRect") {
                navigator_overlay.context2d().fill();
            }
    
        }
        
    }

    let inner_poly;
    let outer_poly;

    if (navigation_type !== "images") {
        outer_poly = [
            [0-1e6, 0-1e6], 
            [0-1e6, image_w_px+1e6], 
            [image_h_px+1e6, image_w_px+1e6],
            [image_h_px+1e6, 0-1e6]
        ];

        let region = annotations[cur_img_name][navigation_type][cur_region_index];

        inner_poly = region;


        overlay.context2d().fillStyle = "#222621";
        overlay.context2d().beginPath();

        for (let poly of [outer_poly, inner_poly]) {

            for (let i = 0; i < poly.length+1; i++) {
                let pt = poly[(i)%poly.length];
                let viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[1], pt[0]));

                if (i == 0) {
                    overlay.context2d().moveTo(viewer_point.x, viewer_point.y);
                }
                else {
                    overlay.context2d().lineTo(viewer_point.x, viewer_point.y);
                }
            }
            overlay.context2d().closePath();

        }
        overlay.context2d().mozFillRule = "evenodd";
        overlay.context2d().fill("evenodd");
    }




    outer_poly = [
        [0-1e6, 0-1e6], 
        [0-1e6, image_w_px+1e6], 
        [image_h_px+1e6, image_w_px+1e6],
        [image_h_px+1e6, 0-1e6]
    ];

    inner_poly = [      
        [cur_tile[0], cur_tile[1]],
        [cur_tile[0], cur_tile[3]],
        [cur_tile[2], cur_tile[3]],
        [cur_tile[2], cur_tile[1]]
    ];

    overlay.context2d().fillStyle = "rgb(0, 0, 0, 0.5)";
    overlay.context2d().beginPath();

    for (let poly of [outer_poly, inner_poly]) {

        for (let i = 0; i < poly.length+1; i++) {
            let pt = poly[(i)%poly.length];
            let viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(pt[1], pt[0]));

            if (i == 0) {
                overlay.context2d().moveTo(viewer_point.x, viewer_point.y);
            }
            else {
                overlay.context2d().lineTo(viewer_point.x, viewer_point.y);
            }
        }
        overlay.context2d().closePath();

    }
    overlay.context2d().mozFillRule = "evenodd";
    overlay.context2d().fill("evenodd");





    let border_viewer_point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(cur_tile[1], cur_tile[0]));
    let border_viewer_point_2 = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(cur_tile[3], cur_tile[2]));
    overlay.context2d().lineWidth = 1;
    overlay.context2d().strokeStyle = "rgb(255, 255, 255, 1.0)";
    overlay.context2d().strokeRect(
        border_viewer_point.x,
        border_viewer_point.y,
        (border_viewer_point_2.x - border_viewer_point.x),
        (border_viewer_point_2.y - border_viewer_point.y)
    );

    let viewer_point = navigator_viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(cur_tile[1], cur_tile[0]));
    let viewer_point_2 = navigator_viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(cur_tile[3], cur_tile[2]));
    navigator_overlay.context2d().lineWidth = 1;
    navigator_overlay.context2d().strokeStyle = "rgb(255, 255, 255, 1.0)";
    navigator_overlay.context2d().strokeRect(
        viewer_point.x,
        viewer_point.y,
        (viewer_point_2.x - viewer_point.x),
        (viewer_point_2.y - viewer_point.y)
    );

    let viewport_bounds = [
        cur_tile[1] / image_w_px,
        (cur_tile[0] / image_h_px) * hw_ratio,
        (cur_tile[3] - cur_tile[1]) / image_w_px,
        ((cur_tile[2] - cur_tile[0]) / image_h_px) * hw_ratio
    ];

    let zoom_bounds = new OpenSeadragon.Rect(
        viewport_bounds[0],
        viewport_bounds[1],
        viewport_bounds[2],
        viewport_bounds[3]
    );
            
    if (!(grid_zoomed)) {
        
        withFastOSDAnimation(viewer.viewport, function() {
            viewer.viewport.fitBounds(zoom_bounds);
        });

        grid_zoomed = true;
    }



    if (gsd != null) {
        let cur_zoom = viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true));
        let measure_width = Math.max(50, 0.08 * container_size.x);
        let measure_width_m = (gsd / cur_zoom) * measure_width;
        let unit;
        let measure_width_metric;
        if (measure_width_m < 1) {
            measure_width_metric = measure_width_m * 100;
            unit = "cm";
        }
        else {
            measure_width_metric = measure_width_m;
            unit = "m";
        }
        let measure_width_text = (Math.ceil(measure_width_metric * 100) / 100).toFixed(2) + " " + unit;


        overlay.context2d().fillStyle = "rgb(255, 255, 255, 0.7)";
        overlay.context2d().fillRect(
            container_size.x - measure_width - 20,
            container_size.y - 30,
            measure_width + 20,
            30
        );
        overlay.context2d().fillStyle = "black";
        overlay.context2d().fillRect(
            container_size.x - measure_width - 10,
            container_size.y - 8,
            measure_width,
            2
        );
        overlay.context2d().fillRect(
            container_size.x - measure_width - 10,
            container_size.y - 10,
            1,
            4
        );
        overlay.context2d().fillRect(
            container_size.x - 10,
            container_size.y - 10,
            1,
            4
        );

        overlay.context2d().fillText(measure_width_text, 
            container_size.x - measure_width - 10,
            container_size.y - 15
        );
    }

    if ((cur_mouse_x != null) && (cur_mouse_y != null)) {

        overlay.context2d().lineWidth = 2;
        overlay.context2d().strokeStyle = overlay_appearance["colors"][cur_edit_layer];
        overlay.context2d().beginPath();
        overlay.context2d().moveTo(0, cur_mouse_y);
        overlay.context2d().lineTo(overlay._containerWidth, cur_mouse_y);
        overlay.context2d().stroke();
        overlay.context2d().closePath();


        overlay.context2d().beginPath();
        overlay.context2d().moveTo(cur_mouse_x, 0);
        overlay.context2d().lineTo(cur_mouse_x, overlay._containerHeight);
        overlay.context2d().stroke();
        overlay.context2d().closePath();
    }
}


function update_grid_overlap_percent() {

    let input_length = ($("#grid_overlap_percent_input").val()).length;
    if ((input_length < 1) || (input_length > 10)) {
        return false;
    }
    let input_val = $("#grid_overlap_percent_input").val();
    if (!(isNumeric(input_val))) {
        return false;
    }
    input_val = parseFloat(input_val);
    if (input_val < 0) {
        return false;
    }

    if (input_val > 0.95) {
        return false;
    }

    return true;

}


function grid_keydown_handler(e) {
    if (e.key === hotkeys["Next Image/Region"]) {
        $("#next_tile_button").click();
    }
    else if (e.key === hotkeys["Previous Image/Region"]) {
        $("#prev_tile_button").click();
    }
    else if (e.key === hotkeys["Save Annotations"]) {
        $("#save_button").click();
    }
}

async function customize_hotkeys() {
    await unselect_selected_annotation(); 
    show_customize_hotkeys_modal();
}


$(window).resize(function() {
    resize_window();
});

window.onoffline = (event) => {
    display_offline_modal();
};