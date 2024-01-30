let count_chart_data;
let count_svg;
let count_xScale;
let count_yScale;
let count_chart_axis;
let count_margin;
let max_count;




function set_count_chart_data() {


    count_chart_data = {};

    let metric = $("#chart_combo").val();

    let slider_val = Number.parseFloat($("#confidence_slider").val());


    let cur_cls_idx = $("#pred_class_select").val();
    let cur_cls;
    if (cur_cls_idx == -1) {
        cur_cls = "All Classes";
    }
    else {
        cur_cls = metadata["object_classes"][cur_cls_idx];
    }


    let navigation_type = $('#navigation_dropdown').val();
    if (metric === "Count" || metric === "Count Per Square Metre") {

        
        if (navigation_type === "images") {
            for (let image_name of Object.keys(annotations)) {
                let nav_item = image_name + "/-1";
                count_chart_data[nav_item] = {"annotation":  0, "prediction": 0};
                for (let i = 0; i < annotations[image_name]["classes"].length; i++) {
                    if (cur_cls_idx == -1 || annotations[image_name]["classes"][i] == cur_cls_idx) {
                        count_chart_data[nav_item]["annotation"]++;
                    }
                }


                if (image_name in predictions) {
                    for (let i = 0; i < predictions[image_name]["scores"].length; i++) {
                        if (cur_cls_idx == -1 || predictions[image_name]["classes"][i] == cur_cls_idx) {
                            if (predictions[image_name]["scores"][i] > slider_val) {
                                count_chart_data[nav_item]["prediction"]++;
                            }
                        }
                    }
                }
            }
        }
        else if (navigation_type === "regions_of_interest") {
            for (let image_name of Object.keys(annotations)) {
                for (let i = 0; i < annotations[image_name]["regions_of_interest"].length; i++) {
                    let nav_item = image_name + "/" + i;
                    count_chart_data[nav_item] = {"annotation":  0, "prediction": 0};
                    for (let j = 0; j < annotations[image_name]["boxes"].length; j++) {
                        let box = annotations[image_name]["boxes"][j];
                        let centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
                        if (point_is_inside_polygon(centre, annotations[image_name]["regions_of_interest"][i])) {
                            if (cur_cls_idx == -1 || annotations[image_name]["classes"][j] == cur_cls_idx) {
                                count_chart_data[nav_item]["annotation"]++;
                            }
                        }
                    }
                    
                    if (image_name in predictions) {
                        for (let j = 0; j < predictions[image_name]["boxes"].length; j++) {
                            if (predictions[image_name]["scores"][j] > slider_val) {
                                let box = predictions[image_name]["boxes"][j];
                                let centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
                                if (point_is_inside_polygon(centre, annotations[image_name]["regions_of_interest"][i])) {
                                    if (cur_cls_idx == -1 || predictions[image_name]["classes"][j] == cur_cls_idx) {
                                        count_chart_data[nav_item]["prediction"]++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else {
            for (let image_name of Object.keys(annotations)) {
                for (let i = 0; i < annotations[image_name][navigation_type].length; i++) {
                    let nav_item = image_name + "/" + i;
                    count_chart_data[nav_item] = {"annotation":  0, "prediction": 0};
                    for (let j = 0; j < annotations[image_name]["boxes"].length; j++) {
                        let box = annotations[image_name]["boxes"][j];
                        let centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
                        if (point_is_inside_box_region(centre, annotations[image_name][navigation_type][i])) {
                            if (cur_cls_idx == -1 || annotations[image_name]["classes"][j] == cur_cls_idx) {
                                count_chart_data[nav_item]["annotation"]++;
                            }
                        }
                    }
                    if (image_name in predictions) {
                        for (let j = 0; j < predictions[image_name]["boxes"].length; j++) {
                            if (predictions[image_name]["scores"][j] > slider_val) {
                                let box = predictions[image_name]["boxes"][j];
                                let centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
                                if (point_is_inside_box_region(centre, annotations[image_name][navigation_type][i])) {
                                    if (cur_cls_idx == -1 || predictions[image_name]["classes"][j] == cur_cls_idx) {
                                        count_chart_data[nav_item]["prediction"]++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        max_count = 0;
        for (let nav_item of Object.keys(count_chart_data)) {
            for (let overlay_name of Object.keys(count_chart_data[nav_item])) {
                let v = count_chart_data[nav_item][overlay_name];
                let image_name = nav_item.split("/")[0];
                let region_index = nav_item.split("/")[1];
                if (metric == "Count Per Square Metre") {
                    let area_m2;
                    if (navigation_type === "images") {

                        let image_height_m = metadata["images"][image_name]["height_px"] * gsd;
                        let image_width_m = metadata["images"][image_name]["width_px"] * gsd;

                        area_m2 = image_width_m * image_height_m;

                    }
                    else if (navigation_type === "regions_of_interest") {
                        let region = annotations[image_name]["regions_of_interest"][region_index];
                        let area_px = get_polygon_area(region);
                        area_m2 = area_px * (gsd ** 2);
                    }
                    else {
                        let region = annotations[image_name][navigation_type][region_index];
                        let area_px = (region[2] - region[0]) * (region[3] - region[1]);
                        area_m2 = area_px * (gsd ** 2);
                    }


                    v = v / area_m2;
                    v = Math.round((v + Number.EPSILON) * 100) / 100;

                    count_chart_data[nav_item][overlay_name] = v;
                }
                if (v > max_count) {
                    max_count = v;
                }
            }
        }
    }

    else if (metric === "Percent Count Error") {

        max_count = 0;
        if (navigation_type === "images") {
            for (let image_name of Object.keys(annotations)) {
                let nav_item = image_name + "/-1";
                count_chart_data[nav_item] = {"annotation":  0, "prediction": 0};
                let annotated_count = 0;
                for (let i = 0; i < annotations[image_name]["classes"].length; i++) {
                    if (cur_cls_idx == -1 || annotations[image_name]["classes"][i] == cur_cls_idx) {
                        annotated_count++;
                    }
                }

                if (image_name in predictions) {
                    let predicted_count = 0;
                    for (let i = 0; i < predictions[image_name]["scores"].length; i++) {
                        if (predictions[image_name]["scores"][i] > slider_val) {
                            if (cur_cls_idx == -1 || predictions[image_name]["classes"][i] == cur_cls_idx) {
                                predicted_count++;
                            }
                        }
                    }
                    if (annotated_count == 0) {
                        count_chart_data[nav_item]["prediction"] = 0;
                    }
                    else {
                        count_chart_data[nav_item]["prediction"] = (Math.abs((predicted_count - annotated_count) / (annotated_count)) * 100);
                    }
                }
                if (count_chart_data[nav_item]["prediction"] > max_count) {
                    max_count = count_chart_data[nav_item]["prediction"];
                }
            }
        }
        else {
            for (let image_name of Object.keys(annotations)) {
                for (let i = 0; i < annotations[image_name][navigation_type].length; i++) {
                    let nav_item = image_name + "/" + i;
                    count_chart_data[nav_item] = {"annotation":  0, "prediction": 0};
                    let annotated_count = 0;

                    for (let j = 0; j < annotations[image_name]["boxes"].length; j++) {

                        let box = annotations[image_name]["boxes"][j];
                        let centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];

                        if (navigation_type === "regions_of_interest") {
                            if (point_is_inside_polygon(centre, annotations[image_name][navigation_type][i])) {
                                if (cur_cls_idx == -1 || annotations[image_name]["classes"][j] == cur_cls_idx) {
                                    annotated_count++;
                                }
                            }
                        }
                        else {
                            if (point_is_inside_box_region(centre, annotations[image_name][navigation_type][i])) {
                                if (cur_cls_idx == -1 || annotations[image_name]["classes"][j] == cur_cls_idx) {
                                    annotated_count++;
                                }
                            }
                        }
                    }
                    if (image_name in predictions) {
                        let predicted_count = 0;
                        for (let j = 0; j < predictions[image_name]["boxes"].length; j++) {
                            if (predictions[image_name]["scores"][j] > slider_val) {
                                let box = predictions[image_name]["boxes"][j];
                                let centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
                                if (navigation_type === "regions_of_interest") {
                                    if (point_is_inside_polygon(centre, annotations[image_name][navigation_type][i])) {
                                        if (cur_cls_idx == -1 || predictions[image_name]["classes"][j] == cur_cls_idx) {
                                            predicted_count++;
                                        }
                                    }

                                }
                                else {
                                    if (point_is_inside_box_region(centre, annotations[image_name][navigation_type][i])) {
                                        if (cur_cls_idx == -1 || predictions[image_name]["classes"][j] == cur_cls_idx) {
                                            predicted_count++;
                                        }
                                    }
                                }
                            }
                        }

                        if (annotated_count == 0) {
                            count_chart_data[nav_item]["prediction"] = 0;
                        }
                        else {
                            count_chart_data[nav_item]["prediction"] = (Math.abs((predicted_count - annotated_count) / (annotated_count)) * 100);
                        }
                    }

                    if (count_chart_data[nav_item]["prediction"] > max_count) {
                        max_count = count_chart_data[nav_item]["prediction"];
                    }
                }
            }
        }
    }
    else {
        if (navigation_type === "images") {
            for (let image_name of Object.keys(annotations)) {
                let image_w = metadata["images"][cur_img_name]["width_px"];
                let image_h = metadata["images"][cur_img_name]["height_px"]
                let fully_annotated_for_training = image_is_fully_annotated_for_training(
                    annotations, 
                    image_name, 
                    image_w,
                    image_h
                );

                let fully_annotated_for_testing = image_is_fully_annotated_for_testing(
                    annotations, 
                    image_name, 
                    image_w,
                    image_h
                );

                if (fully_annotated_for_training || fully_annotated_for_testing) {

                    let nav_item = image_name + "/-1";
                    count_chart_data[nav_item] = {"annotation":  0, "prediction": 0};
                    count_chart_data[nav_item]["annotation"] = 0;

                    let region_key;
                    if (fully_annotated_for_training) {
                        region_key = "training_regions";
                    }
                    else {
                        region_key = "test_regions";
                    }

                    console.log(cur_cls, metric, image_name, region_key);
                    console.log(metrics);

                    count_chart_data[nav_item]["prediction"] = metrics[cur_cls][metric][image_name][region_key][0];
                }
            }
        }
        else {
            for (let image_name of Object.keys(annotations)) {
                for (let i = 0; i < annotations[image_name][navigation_type].length; i++) {
                    let nav_item = image_name + "/" + i;
                    count_chart_data[nav_item] = {"annotation":  0, "prediction": 0};
                    count_chart_data[nav_item]["annotation"] = 0;
                    count_chart_data[nav_item]["prediction"] = metrics[cur_cls][metric][image_name][navigation_type][i];
                }
            }
        }

        let max_100_metrics = [
            "AP (IoU=.50:.05:.95)", 
            "AP (IoU=.50)", 
            "AP (IoU=.75)"
        ];
        let max_1_metrics = [
            "Precision (IoU=.50, conf>.50)",
            "Recall (IoU=.50, conf>.50)",
            "Accuracy (IoU=.50, conf>.50)",
            "F1 Score (IoU=.50, conf>.50)"
        ];

        if (max_100_metrics.includes(metric)) {
            max_count = 100;
        }
        else if (max_1_metrics.includes(metric)) {
            max_count = 1;
        }
        else {
            max_count = 0;
            for (let nav_item of Object.keys(count_chart_data)) {
                if (count_chart_data[nav_item]["prediction"] > max_count) {
                    max_count = count_chart_data[nav_item]["prediction"];
                }
            }
        }
    }
}





function draw_count_chart() {

    let cur_pred_cls_idx = $("#pred_class_select").val();

    let cur_nav_item = cur_img_name + "/" + cur_region_index;
    

    let chart_width = $("#count_chart").width(); // - 10;
    let chart_height = $('#count_chart').height(); // - 10;

    count_margin = 30;

    let num_bars = 2;

    $("#count_chart").empty();
    $("#count_chart").append(`<div id="count_chart_tooltip" class="tooltip" style="z-index: 10"></div>`);


    count_svg = d3.select("#count_chart")
        .append("svg")
        .attr("width", chart_width)
        .attr("height", chart_height);

    let chart = d3.select("#count_chart").select("svg").append("g");


    count_chart_axis = count_svg.append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(" + count_margin + "," + (0.8 * count_margin) + ")");
    

    count_xScale = d3.scaleLinear()
                .domain([0, max_count])
                .range([2.5 * count_margin, chart_width - 1.8 * count_margin]);

    count_yScale = d3.scaleLinear()
                .domain([0, num_bars])
                .range([count_margin, chart_height]);



    count_chart_axis.call(d3.axisTop(count_xScale).ticks(2).tickFormat(d3.format("d")));


    let tooltip = d3.select("#count_chart_tooltip");

    let tip_mouseover = function(d) {
        let cur_nav_item = cur_img_name + "/" + cur_region_index;
        let disp_val = count_chart_data[cur_nav_item][d];
        if (!(Number.isInteger(disp_val))) {
            disp_val = disp_val.toFixed(2);
        }
        let html = numberWithCommas(disp_val);

        tooltip.html(html)
               .style("opacity", 1.0);
        d3.select(this).style("cursor", "default"); 
    }

    let tip_mousemove = function(d) {
        tooltip.style("left", (d3.event.pageX+10) + "px")
               .style("top", (d3.event.pageY-10) + "px");
        d3.select(this).style("cursor", "default"); 

    }

    let tip_mouseleave = function(d) {
        tooltip.style("opacity", 0);
    }

    chart.selectAll("text")
         .data(["Annotated", "Predicted"])
         .enter()
         .append("text")
         .attr("class", "chart_text")
         .attr("x", count_margin * 3)
         .attr("y", function(d, i) {
            return count_margin + 30 * i + 12;
         })
         .attr("alignment-baseline", "central")
         .attr("text-anchor", "end")
         .attr("font-size", "16px")
         .text(function(d) { return d; })
         .style("cursor", "default");

    


    chart.selectAll(".bar")
         .data(Object.keys(count_chart_data[cur_nav_item]))
         .enter()
         .append("rect")
         .attr("class", "bar")
         .attr("id", function (d, i) { return "rect" + i; })
         .attr("x", 3.5 * count_margin)
         .attr("y", function(d, i) {
            return count_margin + 30 * i;
         })
         .attr("width", function(d) {
            return count_xScale(count_chart_data[cur_nav_item][d]) - 2.5 * count_margin;
         })
         .attr("height", 25)
         .attr("fill", function(d) {
            if (cur_pred_cls_idx == -1) {
                return "#222621";
            }
            else {
                return overlay_appearance["colors"][d][cur_pred_cls_idx];
            }
         })
         .attr("stroke", "#ffffff")
         .attr("stroke-width", 1)
         .attr("shape-rendering", "crispEdges")
         .on("mouseover", tip_mouseover)
         .on("mousemove", tip_mousemove)
         .on("mouseleave", tip_mouseleave);


}


function update_count_chart() {

    let cur_pred_cls_idx = $("#pred_class_select").val();

    let cur_nav_item = cur_img_name + "/" + cur_region_index;

    count_xScale.domain([0, max_count]);
    count_chart_axis.transition().duration(250).call(d3.axisTop(count_xScale).ticks(2));

    d3.selectAll(".bar")
        .data(Object.keys(count_chart_data[cur_nav_item]))
        .transition()
        .duration(250)
        .attr("fill", function(d) {
            if (cur_pred_cls_idx == -1) {
                return "#222621";
            }
            else {
                return overlay_appearance["colors"][d][cur_pred_cls_idx];
            }
         })
        .attr("width", function(d) {
            return count_xScale(count_chart_data[cur_nav_item][d]) - 2.5 * count_margin;
        });
}
