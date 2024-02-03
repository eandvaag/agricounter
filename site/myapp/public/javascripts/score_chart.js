
let score_chart_data;
let score_xScale, score_yScale;
let score_chart_x_axis, score_chart_y_axis;
let score_thresholds;
let score_histogram;

function set_score_chart_data() {

    let cur_pred_cls_idx = $("#pred_class_select").val();

    score_thresholds = [];
    for (let i = 0; i < 100; i++) {
        score_thresholds.push(i / 100);
    }

    score_histogram = d3.histogram()
                        .value(function(d) { return d; })
                        .domain([0, 1.0])
                        .thresholds(score_thresholds);

    score_chart_data = {};



    score_chart_data[cur_img_name] = {};
    if (cur_img_name in predictions) {
        score_chart_data[cur_img_name]["scores"] = [];
        let navigation_type = $('#navigation_dropdown').val();
        if (navigation_type === "images") {

            for (let i = 0; i < predictions[cur_img_name]["scores"].length; i++) {
                if (cur_pred_cls_idx == -1 || predictions[cur_img_name]["classes"][i] == cur_pred_cls_idx) {
                    score_chart_data[cur_img_name]["scores"].push(predictions[cur_img_name]["scores"][i]);
                }
            }
        }
        else {
            
            let region = annotations[cur_img_name][navigation_type][cur_region_index];
            for (let i = 0; i < predictions[cur_img_name]["boxes"].length; i++) {
                let box = predictions[cur_img_name]["boxes"][i];
                let centre = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
                if (point_is_inside_polygon(centre, region)) {
                    if (cur_pred_cls_idx == -1 || predictions[cur_img_name]["classes"][i] == cur_pred_cls_idx) {
                        score_chart_data[cur_img_name]["scores"].push(predictions[cur_img_name]["scores"][i]);
                    }
                }
            }
        }
    }
    else {
        score_chart_data[cur_img_name]["scores"] = [];
    }

    let bins = score_histogram(score_chart_data[cur_img_name]["scores"]);
    bins[bins.length-1].x1 = 1.00;

    score_chart_data[cur_img_name]["bins"] = bins;

}

function evaluate_scores(bins, scores) {

    let quality_score = 0;
    let bin_i_prob, bin_i_score;

    if (scores.length > 0) {

        for (let i = 0; i < bins.length; i++) {
            
            bin_i_prob = bins[i].length / scores.length;
            bin_i_score = score_thresholds[i];
            quality_score += bin_i_prob * (1 / (1 + Math.pow(Math.E, -30 * (bin_i_score - 0.80))));
            
        }
    }

    let certainty;
    if (scores.length < 10) {
        certainty = "Low";
    }
    else if (scores.length < 50) {
        certainty = "Moderate";
    }
    else {
        certainty = "High";
    }

    return [quality_score, certainty];
}


function draw_score_chart() {


    let cur_pred_cls_idx = $("#pred_class_select").val();

    let chart_color;
    if (cur_pred_cls_idx == -1) {
        chart_color = "#222621";
    }
    else {
        chart_color = overlay_appearance["colors"]["prediction"][cur_pred_cls_idx];
    }

    $("#score_chart").empty();

    let chart_width = $("#score_chart").width();
    let chart_height = $("#score_chart").height();

    let margin_top = 10;
    let margin_right = 4;
    let margin_bottom = 20;
    let margin_left = 45;

    let width = chart_width - margin_left - margin_right;
    let height = chart_height - margin_top - margin_bottom;

    score_svg = d3.select("#score_chart")
                    .append("svg")
                    .attr("id", "score_chart_svg")
                    .attr("width", chart_width)
                    .attr("height", chart_height)
                    .append("g")
                    .attr("transform",
                      "translate(" + margin_left + "," + margin_top + ")");

    score_xScale = d3.scaleLinear()
                        .domain([0.25, 1])
                        .range([0, width]);

    score_chart_x_axis = score_svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")");
    
    score_chart_x_axis.call(d3.axisBottom(score_xScale).tickSizeOuter(0).tickValues([0.25, 0.5, 0.75, 1.0]) //.tickFormat(d3.format("d"))); //x => `${x.toFixed(2)}`));
    .tickFormat((d, i) => ['0.25', '0.5', '0.75', '1'][i])); 


    let bins = score_chart_data[cur_img_name]["bins"];
    let scores = score_chart_data[cur_img_name]["scores"];

    let ret = evaluate_scores(bins, scores);
    let quality_score = Math.round((ret[0] + Number.EPSILON) * 100);
    let certainty = ret[1];

    $("#quality_score").html(quality_score + "% (" + certainty + " Certainty)");

    score_yScale = d3.scaleLinear()
                .range([height, 0]);

    score_yScale.domain([0, d3.max(bins, function(d) { return d.length; })]);

    score_chart_y_axis = score_svg.append("g")
                                .attr("class", "y axis");

    score_chart_y_axis.call(d3.axisLeft(score_yScale).ticks(4));


    score_svg.selectAll(".score_rect")
            .data(bins)
            .enter()
            .append("rect")
            .attr("class", "score_rect")
            .attr("x", 1)
            .attr("transform", function(d) {
                let y_trans = scores.length > 0 ? score_yScale(d.length) : height;
                return "translate(" + score_xScale(d.x0) + "," + y_trans + ")"; 
            })
            .attr("width", function(d) { return score_xScale(d.x1) - score_xScale(d.x0); })
            .attr("height", function(d) {
                let y_trans = scores.length > 0 ? score_yScale(d.length) : height;
                return height - y_trans;
            })
            .style("fill", chart_color)
            .attr("shape-rendering", "crispEdges")
            .attr("stroke", "white")
            .attr("stroke-width", "1");
    
    
    let slider_val = parseFloat($("#confidence_slider").val());
    score_svg
        .append("line")
        .attr("class", "score_line")
        .attr("x1", score_xScale(slider_val) )
        .attr("x2", score_xScale(slider_val) )
        .attr("y1", 0)
        .attr("y2", height)
        .attr("stroke", "white")
        .attr("stroke-dasharray", "4")
        .attr("stroke-width", "1");

}


function update_score_chart() {

    let cur_pred_cls_idx = $("#pred_class_select").val();

    let chart_height = $('#score_chart').height();

    let margin_top = 10;
    let margin_bottom = 20;

    let height = chart_height - margin_top - margin_bottom;


    let chart_color;
    if (cur_pred_cls_idx == -1) {
        chart_color = "#222621";
    }
    else {
        chart_color = overlay_appearance["colors"]["prediction"][cur_pred_cls_idx];
    }

    let bins = score_chart_data[cur_img_name]["bins"];
    let scores = score_chart_data[cur_img_name]["scores"];


    let ret = evaluate_scores(bins, scores);
    let quality_score =  Math.round((ret[0] + Number.EPSILON) * 100);
    let certainty = ret[1];

    $("#quality_score").html(quality_score + "% (" + certainty + " Certainty)");

    score_yScale.domain([0, d3.max(bins, function(d) { return d.length; })]);

    score_chart_y_axis.transition().duration(250).call(d3.axisLeft(score_yScale).ticks(4));

    d3.selectAll(".score_rect")
        .data(bins)
        .transition()
        .duration(250)
        .attr("transform", function(d) {
            let y_trans = scores.length > 0 ? score_yScale(d.length) : height;
            return "translate(" + score_xScale(d.x0) + "," + y_trans + ")"; 
        })
        .attr("height", function(d) {
            let y_trans = scores.length > 0 ? score_yScale(d.length) : height;
            return height - y_trans;
        })
        .style("fill", chart_color);

    let slider_val = parseFloat($("#confidence_slider").val());
    d3.selectAll(".score_line")
        .data([slider_val])
        .transition()
        .duration(250)
        .attr("x1", score_xScale(slider_val))
        .attr("x2", score_xScale(slider_val));

}