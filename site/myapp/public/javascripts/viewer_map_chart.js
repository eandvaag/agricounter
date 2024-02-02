let wheat = [245, 222, 179];
let forestgreen = [34, 139,  34];


function range_map(old_val, old_min, old_max, new_min, new_max) {
    new_val = (((old_val - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min;
    return new_val;
}

function color_map(num, min_num, max_num, c1, c2) {
    let fraction = (num - min_num) / (max_num - min_num);
    let r = ((c2[0] - c1[0]) * fraction) + c1[0];
    let g = ((c2[1] - c1[1]) * fraction) + c1[1];
    let b = ((c2[2] - c1[2]) * fraction) + c1[2];

    return [r, g, b]
}


function draw_map_chart() {

    let num_completed = 0;
    for (let image_name of Object.keys(annotations)) {

        let image_width_px = metadata["images"][image_name]["width_px"];
        let image_height_px = metadata["images"][image_name]["height_px"];
        if (image_is_fully_annotated(annotations, image_name, image_width_px, image_height_px)) {
            num_completed++;
        }
    }

    let include_annotated_map = num_completed >= 3;

    let chart_width = (($("#chart_container").width() / 2) - 64) + "px";
    let chart_height = chart_width;

    $("#chart_container").empty();

    $("#chart_container").append(
        `<table class="transparent_table"><tr id="map_row"></tr></table>`);

    if (include_annotated_map) {

        $("#map_row").append(
            `<td>` +
                `<div id="map_container" style="height: ${chart_height}; width: ${chart_width};">` +
                    `<div id="map_chart_tooltip" class="tooltip" style="z-index: 10"></div>` +
                `</div>` +
            `</td>` +
            `<td>` +
                `<div id="legend_container" style="height: ${chart_height}; width: 60px"></div>` +
            `</td>` +
            `<td>` +
                `<div id="pred_map_container" style="height: ${chart_height}; width: ${chart_width};">` +
                    `<div id="pred_map_chart_tooltip" class="tooltip" style="z-index: 10"></div>` +
                `</div>` +
            `</td>` 
        );

    }
    else {

        $("#map_row").append(
            `<td>` +
                `<div id="pred_map_container" style="height: ${chart_height}; width: ${chart_width};">` +
                    `<div id="pred_map_chart_tooltip" class="tooltip" style="z-index: 10"></div>` +
                `</div>` +
            `</td>`  +
            `<td>` +
                `<div id="legend_container" style="height: ${chart_height}; width: 60px"></div>` +
            `</td>`

        );
    }

    chart_width = $("#pred_map_container").width();
    chart_height = $("#pred_map_container").height();

    let svg;
    if (include_annotated_map) {
        svg = d3.select("#map_container")
            .append("svg")
            .attr("width", chart_width)
            .attr("height", chart_height);
    }
    let pred_svg = d3.select("#pred_map_container")
        .append("svg")
        .attr("width", chart_width)
        .attr("height", chart_height);



    if (pred_map_url !== null) {
        let chart;
        if (include_annotated_map) {
            chart = d3.select("#map_container").select("svg").append("g");
        }
        let pred_chart = d3.select("#pred_map_container").select("svg").append("g");
            
        let margin = 110;


        let circle_data = [];
        let max_latitude = -10000;
        let min_latitude = 10000;
        let max_longitude = -10000;
        let min_longitude = 10000;
        
        for (let dzi_image_path of dzi_image_paths) {
            let image_name = basename(dzi_image_path)
            image_name = image_name.substring(0, image_name.length - 4);

            let latitude = metadata["images"][image_name]["latitude"];
            let longitude = metadata["images"][image_name]["longitude"];
            let image_width_px = metadata["images"][image_name]["width_px"];
            let image_height_px = metadata["images"][image_name]["height_px"];

            let color;

            if (image_is_fully_annotated_for_fine_tuning(annotations, image_name, image_width_px, image_height_px)) {
                color = overlay_appearance["colors"]["fine_tuning_region"];
            }
            else if (image_is_fully_annotated_for_testing(annotations, image_name, image_width_px, image_height_px)) {
                color = overlay_appearance["colors"]["test_region"];
            }
            else {
                color = "white";
            }

            circle_data.push({
                "latitude": latitude,
                "longitude": longitude,
                "color": color,
                "image_name": image_name,
                "dzi_image_path": dzi_image_path
            });

            if (latitude < min_latitude) {
                min_latitude = latitude;
            }
            if (latitude > max_latitude) {
                max_latitude = latitude;
            }
            if (longitude < min_longitude) {
                min_longitude = longitude;
            }
            if (longitude > max_longitude) {
                max_longitude = longitude;
            }
        }
        let chart_x_axis;
        let chart_y_axis;
        let pred_chart_x_axis;
        let pred_chart_y_axis;
        if (include_annotated_map) {
            chart_x_axis = svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + (chart_height - (margin / 2)) + ")");

            chart_y_axis = svg.append("g")
                    .attr("class", "y axis")
                    .attr("transform", "translate(" + (margin / 2) + ", 0)");
        }


        pred_chart_x_axis = pred_svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + (chart_height - (margin / 2)) + ")");

        pred_chart_y_axis = pred_svg.append("g")
                .attr("class", "y axis")
                .attr("transform", "translate(" + (margin / 2) + ", 0)");

                
        xScale = d3.scaleLinear()
            .domain([min_longitude, max_longitude])
            .range([margin, chart_width - margin]);

        yScale = d3.scaleLinear()
            .domain([min_latitude, max_latitude])
            .range([chart_height - margin, margin]);

        if (include_annotated_map) {
            chart_x_axis.call(d3.axisBottom(xScale).tickValues([min_longitude, max_longitude]).tickFormat(x => `${x.toFixed(4)}`));
            chart_y_axis.call(d3.axisLeft(yScale).tickValues([min_latitude, max_latitude]).tickFormat(x => `${x.toFixed(4)}`));
        }
        else {
            pred_chart_y_axis.call(d3.axisLeft(yScale).tickValues([min_latitude, max_latitude]).tickFormat(x => `${x.toFixed(4)}`));
        }
        pred_chart_x_axis.call(d3.axisBottom(xScale).tickValues([min_longitude, max_longitude]).tickFormat(x => `${x.toFixed(4)}`));
        
        

        if (include_annotated_map) {
            svg.append("text")
                .attr("text-anchor", "middle")
                .attr("x", - (chart_height / 2))
                .attr("y", (margin / 2) - 30)
                .attr("dy", ".75em")
                .attr("transform", "rotate(-90)")
                .text("Latitude");

            svg.append("text")
                .attr("text-anchor", "middle")
                .attr("x", ((chart_width / 2)))
                .attr("y", chart_height - (margin / 2) + 30)
                .text("Longitude");
        }
        else {
            pred_svg.append("text")
                .attr("text-anchor", "middle")
                .attr("x", - (chart_height / 2))
                .attr("y", (margin / 2) - 30)
                .attr("dy", ".75em")
                .attr("transform", "rotate(-90)")
                .text("Latitude");
        }

        pred_svg.append("text")
            .attr("text-anchor", "middle")
            .attr("x", ((chart_width / 2)))
            .attr("y", chart_height - (margin / 2) + 30)
            .text("Longitude");


        if (include_annotated_map) {
            chart.selectAll("text")
                .data(["Number of Annotated Seedlings Per Square Metre"])
                .enter()
                .append("text")
                .attr("x", chart_width / 2)
                .attr("y", margin / 2)
                .attr("alignment-baseline", "central")
                .attr("text-anchor", "middle")
                .attr("font-size", "20px")
                .text(function(d) {
                    return d;
                });
        }
        let pred_text = "Number of Predicted Seedlings Per Square Metre";

        pred_chart.selectAll("text")
            .data([pred_text])
            .enter()
            .append("text")
            .attr("x", chart_width / 2)
            .attr("y", margin / 2)
            .attr("alignment-baseline", "central")
            .attr("text-anchor", "middle")
            .attr("font-size", "20px")
            .text(function(d) {
                return d;
            });

        if (include_annotated_map) {
            chart.append("svg:image")
                .attr("x", margin)
                .attr("y", margin)
                .attr("width", chart_width - 2 * margin)
                .attr("height", chart_height - 2 * margin)
                .attr("xlink:href", map_url);
        }
        pred_chart.append("svg:image")
            .attr("x", margin)
            .attr("y", margin)
            .attr("width", (chart_width - 2 * margin))
            .attr("height", chart_height - 2 * margin)
            .attr("xlink:href", pred_map_url);

        let vmin;
        let vmax;
        if (min_max_rec !== null) {
            vmin = min_max_rec["vmin"];
            vmax = min_max_rec["vmax"];
        

            let legend_svg = d3.select("#legend_container")
                                .append("svg")
                                .attr("width", "60px")
                                .attr("height", chart_height);

            let cmap = d3.select("#legend_container").select("svg").append("g");


            let min_color = wheat;
            let max_color = forestgreen;

            let rects = [];
            
            let num_rects = 1000;
            for (let i = 0; i < num_rects; i++) {

                let v = range_map(i, 0, num_rects, vmin, vmax);
                let c = color_map(v, vmin, vmax, min_color, max_color);
                
                rects.push({
                    "color": "rgb(" + c[0] + ", " + c[1] + ", " + c[2] + ")",
                    "v": v
                });
            }

            let legend_yScale = d3.scaleLinear()
                .domain([vmin, vmax])
                .range([chart_height - margin, margin]);


            let legend_y_axis = legend_svg.append("g")
                .attr("class", "map_legend axis")
                .attr("transform", "translate(" + 60 + ", 0)");

            legend_y_axis.call(d3.axisLeft(legend_yScale).tickValues([vmin, vmax]).tickFormat(d3.format("d")).tickSize(25));

            cmap.selectAll(".rect")
            .data(rects)
            .enter()
            .append("rect")
            .attr("x", 40)
            .attr("y", function(d, i) {
                return legend_yScale(d.v);
            })
            .attr("height", function(d, i) {
                if (i == num_rects -1) {
                    return (legend_yScale(d.v) - legend_yScale(vmax)) + 1;
                }
                else {
                    return (legend_yScale(d.v) - legend_yScale(rects[i+1]["v"])) + 1;
                }
            })
            .attr("width", 20)
            .attr("fill", function(d) {
                return d["color"];
            });
        }



        if (include_annotated_map) {
            let tooltip = d3.select("#map_chart_tooltip");
            let tip_mouseover = function(d) {

                $("#map_chart_tooltip").show();
                let html = "Image: " + d.image_name;

                tooltip.html(html);
                let tooltip_width = $("#map_chart_tooltip").width();
                tooltip.style("opacity", 1.0)
                    .style("left", (d3.event.pageX - (tooltip_width / 2)) + "px")
                    .style("top", (d3.event.pageY - 40) + "px");
                d3.select(this).style("cursor", "pointer"); 
            }

            let tip_mouseleave = function(d) {
                tooltip.style("opacity", 0);
                $("#map_chart_tooltip").hide();
                d3.select(this).style("cursor", "default"); 
            }

        
            chart.selectAll("circle")
                .data(circle_data)
                .enter()
                .append("circle")
                .attr("cx", function(d) {
                    return xScale(d["longitude"]);
                })
                .attr("cy", function(d) {
                    return yScale(d["latitude"]);
                })
                .attr("r", 5)
                .attr("fill", function(d) {
                    return d["color"];
                })
                .attr("stroke", "black")
                .attr("stroke-width", 1)

                .on("click", function(d) {
                    $("#navigation_dropdown").val("images");
                    create_navigation_table();
                    update_count_combo(true);
                    cur_region_index = -1;
                    show_image(d["image_name"]);
                })
                .on("mouseover", tip_mouseover)
                .on("mouseleave", tip_mouseleave);

        }
        
        let pred_tooltip = d3.select("#pred_map_chart_tooltip");
        

        let pred_tip_mouseover = function(d) {

            $("#pred_map_chart_tooltip").show();
            let html = "Image: " + d.image_name;

            pred_tooltip.html(html);
            let tooltip_width = $("#pred_map_chart_tooltip").width();
            pred_tooltip.style("opacity", 1.0)
                .style("left", (d3.event.pageX - (tooltip_width / 2)) + "px")
                .style("top", (d3.event.pageY - 40) + "px");
            d3.select(this).style("cursor", "pointer"); 
        }

        let pred_tip_mouseleave = function(d) {
            pred_tooltip.style("opacity", 0);
            $("#pred_map_chart_tooltip").hide();
            d3.select(this).style("cursor", "default"); 
        }

        pred_chart.selectAll("circle")
            .data(circle_data)
            .enter()
            .append("circle")
            .attr("cx", function(d) {
                return xScale(d["longitude"]);
            })
            .attr("cy", function(d) {
                return yScale(d["latitude"]);
            })
            .attr("r", 5)
            .attr("fill", function(d) {
                return d["color"];
            })
            .attr("stroke", "black")
            .attr("stroke-width", 1)

            .on("click", function(d) {
                $("#navigation_dropdown").val("images");
                create_navigation_table();
                update_count_combo(true);
                cur_region_index = -1;
                show_image(d["image_name"]);
            })
            .on("mouseover", pred_tip_mouseover)
            .on("mouseleave", pred_tip_mouseleave);         

    }
}

