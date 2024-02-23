let ridgeline_svg;

let ridgeline_xScale;
let ridgeline_chart_x_axis;

let ridgeline_chart_y_axis;
let yName;

let all_bins;


function show_ridgeline_modal() {

    show_modal_message("Confidence Distributions",
     `<table>` +
         `<tr>` +
             `<td>` +
                 `<div class="header2">Sort By</div>` +
             `</td>` +
             `<td>` +
                 `<select id="ridgeline_sort_combo" style="width: 200px; margin-left: 10px" class="nonfixed_dropdown">` +
                    `<option value="quality_score" selected>Quality Score</option>` +
                    `<option value="image_name">Image Name</option>` +
                 `</select>` +
             `</td>` +
         `</tr>` +
     `</table>` +
 
     `<div style="width: 850px; text-align: center">` +
        `<div id="ridgeline_loading"><div style="height: 30px"></div><div class="loader"></div></div>` +
         `<div style="width: 800px; height: 500px" class="scrollable_area" id="ridgeline_content" hidden>` +
             `<div id="ridgeline_chart" style="margin: 0 auto; border: none; width: 775px; height: 450px"></div>` +
         `</div>` +
     `</div>` 
    , modal_width=850, display=true);

    $("#ridgeline_sort_combo").change(function() {
        draw_ridgeline_chart();
    });

    let image_list = [];
    for (let image_name of Object.keys(annotations)) {
        if (!(image_name in predictions)) {
            image_list.push(image_name);
        }
    }
    $.post($(location).attr('href'),
    {
        action: "retrieve_predictions",
        image_names: image_list.join(",")
    },
    function(response, status) {

        if (response.error) {
            show_modal_message("Error", response.message);
        }
        else {
            for (let image_name of image_list) {
                if (image_name in response.predictions) {
                    predictions[image_name] = response.predictions[image_name];
                }
            }
            $("#ridgeline_loading").hide();
            $("#ridgeline_content").show();
            draw_ridgeline_chart();
        }
    });
    //  draw_ridgeline_chart();
 }



function draw_ridgeline_chart() {

    let cur_pred_cls_idx = $("#pred_class_select").val();
    $("#ridgeline_chart").empty();

    let image_names;
    let image_name_to_quality = {};

    let image_qualities = [];
    for (let image_name of Object.keys(predictions)) {
        let scores = [];
        for (let i = 0; i < predictions[image_name]["scores"].length; i++) {
            if (cur_pred_cls_idx == -1 || predictions[image_name]["classes"][i] == cur_pred_cls_idx) {
                scores.push(predictions[image_name]["scores"][i]);
            }
        }
        let bins = score_histogram(scores);
        bins[bins.length-1].x1 = 1.00;
        let quality_score = evaluate_scores(bins, scores)[0];
        image_qualities.push([image_name, quality_score]);
        image_name_to_quality[image_name] = quality_score;
    }

    image_qualities.sort((a, b) => {
        if (a[1] < b[1]) {
            return -1;
        }
        if (a[1] > b[1]) {
            return 1;
        }
        return 0;
    });

    if ($("#ridgeline_sort_combo").val() === "image_name") {
        image_names = natsort(Object.keys(predictions));
    }
    else {
        image_names = [];
        for (let i = 0; i < image_qualities.length; i++) {
            image_names.push(image_qualities[i][0]);
        }

    }

    let ticks = [];
    for (let i = 20; i <= 105; i += 0.1 ) {
        ticks.push(i);
    }

    let kde = kernelDensityEstimator(kernelEpanechnikov(1.0), ticks);

    let all_density = [];
    for (let image_name of image_names) {
        if (predictions[image_name]["scores"].length == 0) {
            let density_vals = [];
            for (let i = 20; i <= 105; i += 0.1 ) {
                density_vals.push([i, 0]);
            }
            all_density.push({image_name, density: density_vals});
        }
        else {
            let scores = []
            for (let score of predictions[image_name]["scores"]) {
                scores.push(score * 100);
            }

            let density = kde(scores);
            all_density.push({image_name: image_name, density: density});
        }
    }

    let max_density = 0;
    for (let density of all_density) {
        for (let i = 0; i < density.density.length; i++) {
            if (density.density[i][1] > max_density) {
                max_density = density.density[i][1];
            }
        }
    }

    let disp_image_names = [];
    for (let image_name of image_names) {
        if (image_name.length > 25) {
            disp_image_names.push(image_name.substring(0, 10) + "..." + image_name.substring(image_name.length-10, image_name.length));
        }
        else {
            disp_image_names.push(image_name);
        }
    }
    let max_name_width = get_max_name_width(disp_image_names, "normal 12px sans-serif");

    
    let margin_top = 60;
    let margin_right = 60;
    let margin_bottom = 40;
    let margin_left = max_name_width + 20;


    let min_row_spacing = 40;

    let new_chart_height = Math.max(image_names.length * min_row_spacing, 350);

    $("#ridgeline_chart").height(new_chart_height);

    let chart_width = $("#ridgeline_chart").width();
    let chart_height = $("#ridgeline_chart").height();

    let width = chart_width - margin_left - margin_right;
    let height = chart_height - margin_top - margin_bottom;

    ridgeline_svg = d3.select("#ridgeline_chart")
                    .append("svg")
                    .attr("id", "ridgeline_chart_svg")
                    .attr("width", chart_width)
                    .attr("height", chart_height)
                    .append("g")
                    .attr("transform",
                    "translate(" + margin_left + "," + margin_top + ")");


    ridgeline_xScale = d3.scaleLinear()
                    .domain([20, 105])
                    .range([0, width]);

    ridgeline_chart_x_axis = ridgeline_svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height  + ")");

    ridgeline_chart_x_axis.call(d3.axisBottom(ridgeline_xScale).tickSizeOuter(0)
        .tickValues([25, 50, 75, 100])
        .tickFormat((d, i) => ['0.25', '0.5', '0.75', '1'][i]));

    let max_scores_length = 0;
    for (let image_name of image_names) {
        if (predictions[image_name]["scores"].length > max_scores_length) {
            max_scores_length = predictions[image_name]["scores"].length;
        }
    }

    yName = d3.scaleBand()
            .domain(image_names)
            .range([0, height])
            .paddingInner(1);

    ridgeline_chart_y_axis = ridgeline_svg.append("g")
        .attr("class", "y axis")
        
    ridgeline_chart_y_axis.call(d3.axisLeft(yName)
    .tickValues(image_names)
    .tickFormat((d, i) => disp_image_names[i]));
    
    ridgeline_yScale = d3.scaleLinear()
                        .domain([0, max_density])
                        .range([min_row_spacing*1.5, 0]);

    let color;
    if (cur_pred_cls_idx == -1) {
        color = "#222621";
    }
    else {
        color = overlay_appearance["colors"]["prediction"][cur_pred_cls_idx];
    }

    let ridgeline_chart = d3.select("#ridgeline_chart").select("svg").append("g");

    ridgeline_chart.selectAll("text")
        .data(image_names)
        .enter()
        .append("text")
        .attr("class", "chart_text")
        .attr("x", chart_width - margin_right + 3)
        .attr("y", function(d, i) {
            
            return yName(d) + (1.5*min_row_spacing) - 1;
        })
        .attr("alignment-baseline", "central")
        .attr("text-anchor", "start")
        .attr("font-size", "14px")
        .text(function(d) { 
            return Math.round((image_name_to_quality[d] + Number.EPSILON) * 100) + "%";
        })
        .style("cursor", "default");

    
    ridgeline_svg.selectAll("areas")
        .data(all_density)
        .enter()
        .append("path")
        .attr("class", "chart_area")
        .attr("transform", function(d) {
            return ("translate(0," + (yName(d.image_name) - (1.5*min_row_spacing)) + ")" );
        })
        .datum(function(d) {
            return d.density;
        })
        .attr("fill", color)
        .attr("stroke", "white")
        .attr("stroke-width", 1)
        .attr("d", 
            d3.line()
            .curve(d3.curveBasis)
            .x(function(d) { return ridgeline_xScale(d[0]); })
            .y(function(d) { return ridgeline_yScale(d[1]); })
        )
        .attr("cursor", "pointer")
        .on("mouseover", handleMouseOver)
        .on("mouseout", handleMouseOut)
        .on("click", function(d, i) {
            $("#navigation_dropdown").val("images").change();
            change_image(image_names[i] + "/-1");
            close_modal();
         });


    ridgeline_svg.selectAll(".y.axis .tick").style("cursor", "pointer");
    ridgeline_svg.selectAll(".y.axis .tick").on("mouseover", function(d, i) { 
        ridgeline_svg.selectAll(".chart_area")
        .filter(function(e, j) { return i == j; })
        .attr("fill", "white");
        $(this).children('text').css("font-weight", "bold"); 
    });
    ridgeline_svg.selectAll(".y.axis .tick").on("mouseout", function(d, i) { 
        ridgeline_svg.selectAll(".chart_area")
        .filter(function(e, j) { return i == j; })
        .attr("fill", color);
        
        $(this).children('text').css("font-weight", "normal"); 
    });
    ridgeline_svg.selectAll(".y.axis .tick").on("click", function(d, i) { 
        $("#navigation_dropdown").val("images").change();
        change_image(image_names[i] + "/-1");
        close_modal();
    });

    function handleMouseOver(d, i) {
        d3.select(this).attr("fill", "white");

        let tick_element = ridgeline_svg.selectAll(".y.axis .tick")._groups[0][i];
        $(tick_element).children('text').css("font-weight", "bold");

    }


    function handleMouseOut(d, i) {
        d3.select(this).attr("fill", color);
        let tick_element = ridgeline_svg.selectAll(".y.axis .tick")._groups[0][i];
        $(tick_element).children('text').css("font-weight", "normal");
    }

}

function kernelDensityEstimator(kernel, X) {
    return function(V) {
        return X.map(function(x) {
        return [x, d3.mean(V, function(v) { return kernel(x - v); })];
        });
    };
}
function kernelEpanechnikov(k) {
    return function(v) {
        return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
    };
}
  