
let anno_cls_breakdown_xScale;
let tooltip_text;

function get_anno_cls_breakdown_data() {

    let anno_cls_counts = [];
    for (let i = 0; i < metadata["object_classes"].length; i++) {
        anno_cls_counts.push(0);
    }

    for (let image_name of Object.keys(annotations)) {
        for (let cls of annotations[image_name]["classes"]) {
            anno_cls_counts[cls]++;
        }
    }

    let anno_cls_breakdown_data = [];
    anno_cls_breakdown_data.push({
        "start": 0,
        "end": anno_cls_counts[0]
    });
    for (let i = 1; i < metadata["object_classes"].length; i++) {
        let v = {
            "start": anno_cls_breakdown_data[i-1]["end"],
            "end": anno_cls_breakdown_data[i-1]["end"] + anno_cls_counts[i]
        }
        anno_cls_breakdown_data.push(v);
    }
    
    return anno_cls_breakdown_data;
}


function set_tooltip_text(anno_cls_breakdown_data) {
    let max_text_width = 0;
    for (let i = 0; i < metadata["object_classes"].length; i++) {
        let text_width = get_text_width(metadata["object_classes"][i], "13px sans-serif");
        if (text_width > max_text_width) {
            max_text_width = text_width;
        }
    }
    let obj_col_width = max_text_width + 5;

    tooltip_text = ``;
    for (let i = 0; i < anno_cls_breakdown_data.length; i++) {
        let v  = anno_cls_breakdown_data[i]["end"] - anno_cls_breakdown_data[i]["start"];
        tooltip_text += `<table>` +
                            `<tr>` +
                                `<td>` + 
                                    `<div style="width: ` + obj_col_width + `px; text-align: left">` + metadata["object_classes"][i] + `</div>` +
                                `</td>` +
                                `<td>` + 
                                    `<div style="width: 70px; text-align: right">` + v + `</div>` +
                                `</td>` +
                            `</tr>` +
                        `</table`;
    }
}

function draw_cls_breakdown_chart() {

    if (metadata["object_classes"].length == 1) {
        return;
    }

    let anno_cls_breakdown_data = get_anno_cls_breakdown_data();

    let chart_width = $("#anno_cls_breakdown_chart").width();
    let chart_height = $('#anno_cls_breakdown_chart').height();

    $("#anno_cls_breakdown_chart").empty();
    $("#anno_cls_breakdown_chart").append(`<div id="anno_cls_breakdown_chart_tooltip" class="tooltip" style="z-index: 10"></div>`);
    $("#anno_cls_breakdown_container").css("border", "1px solid white");

    d3.select("#anno_cls_breakdown_chart")
        .append("svg")
        .attr("width", chart_width)
        .attr("height", chart_height);

    let chart = d3.select("#anno_cls_breakdown_chart").select("svg").append("g");

    let total_count = anno_cls_breakdown_data[anno_cls_breakdown_data.length-1]["end"];
    anno_cls_breakdown_xScale = d3.scaleLinear()
                .domain([0, total_count])
                .range([0, chart_width]);

    let anno_cls_breakdown_yScale = d3.scaleLinear()
                .domain([0, 1])
                .range([0, chart_height]);

    let tooltip = d3.select("#anno_cls_breakdown_chart_tooltip");

    let tip_mouseover = function() {

        tooltip.html(tooltip_text)
                .style("opacity", 1.0);
        d3.select(this).style("cursor", "default");
    }

    let tip_mousemove = function(d) {
        let h = $("#anno_cls_breakdown_chart_tooltip").height();
        let w = $("#anno_cls_breakdown_chart_tooltip").width();
        tooltip.style("left", (d3.event.pageX-w) + "px")
                .style("top", (d3.event.pageY-h-10) + "px");
        d3.select(this).style("cursor", "default"); 
    }

    let tip_mouseleave = function(d) {
        tooltip.style("opacity", 0)
               .style("left", (-1000) + "px")
               .style("top", (-1000) + "px");
    }
            
    set_tooltip_text(anno_cls_breakdown_data);

    chart.selectAll(".breakdown_bar")
         .data(anno_cls_breakdown_data)
         .enter()
         .append("rect")
         .attr("class", "breakdown_bar")
         .attr("x", function(d) {
            return anno_cls_breakdown_xScale(d["start"]);
         })
         .attr("y", anno_cls_breakdown_yScale(0))
         .attr("width", function(d) {
            return anno_cls_breakdown_xScale(d["end"] - d["start"]);
         })
         .attr("height", anno_cls_breakdown_yScale(1))
         .attr("fill", function(d, i) {
            return overlay_appearance["colors"]["annotation"][i];
         })
         .on("mouseover", tip_mouseover)
         .on("mousemove", tip_mousemove)
         .on("mouseleave", tip_mouseleave);

}


function update_cls_breakdown_chart() {

    if (metadata["object_classes"].length == 1) {
        return;
    }
    let anno_cls_breakdown_data = get_anno_cls_breakdown_data();
    let total_count = anno_cls_breakdown_data[anno_cls_breakdown_data.length-1]["end"];
    anno_cls_breakdown_xScale.domain([0, total_count]);

    set_tooltip_text(anno_cls_breakdown_data);
    d3.selectAll(".breakdown_bar")
        .data(anno_cls_breakdown_data)
        .transition()
        .duration(250)
        .attr("x", function(d) {
            return anno_cls_breakdown_xScale(d["start"]);
        })
        .attr("width", function(d) {
            return anno_cls_breakdown_xScale(d["end"] - d["start"]);
        })
        .attr("fill", function(d, i) {
            return overlay_appearance["colors"]["annotation"][i];
        });
}
