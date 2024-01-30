
/*
    works with y, x coords
*/
function point_is_inside_polygon(point, vs) {
    // ray-casting algorithm based on
    // https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    
    var x = point[0], y = point[1];
    
    var inside = false;
    for (var i = 0, j = vs.length - 1; i < vs.length; j = i++) {
        var xi = vs[i][0], yi = vs[i][1];
        var xj = vs[j][0], yj = vs[j][1];
        
        var intersect = ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    
    return inside;
};

/*
    uses y, x coords
*/
function get_polygon_area(vertices) {
    var total = 0;

    for (var i = 0, l = vertices.length; i < l; i++) {
      var addX = vertices[i][1]; //.x;
      var addY = vertices[i == vertices.length - 1 ? 0 : i + 1][0]; //.y;
      var subX = vertices[i == vertices.length - 1 ? 0 : i + 1][1]; //.x;
      var subY = vertices[i][0]; //.y;

      total += (addX * addY * 0.5);
      total -= (subX * subY * 0.5);
    }

    return Math.abs(total);
}
  
/*
    uses y, x coords
*/
function get_bounding_box_for_polygon(poly) {
    let min_y, min_x, max_y, max_x;
    for (let i = 0; i < poly.length; i++) {
        if (i == 0) {
            min_y = poly[i][0];
            min_x = poly[i][1];
            max_y = poly[i][0];
            max_x = poly[i][1];
        }
        else {

            if (poly[i][0] < min_y) {
                min_y = poly[i][0];
            }
            if (poly[i][1] < min_x) {
                min_x = poly[i][1];
            }
            if (poly[i][0] > max_y) {
                max_y = poly[i][0];
            }
            if (poly[i][1] > max_x) {
                max_x = poly[i][1];
            }
        }
    }

    return [min_y, min_x, max_y, max_x];
}




// https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#JavaScript
/*
    accepts x, y coordinates
*/
function clip_polygons(subjectPolygon, clipPolygon) {
    let flip_subjectPolygon;
    let flip_clipPolygon;

    // if (coord_format === "xy") {
    // }

    flip_subjectPolygon = [];
    for (let c of subjectPolygon) {
        flip_subjectPolygon.push([c[0], c[1]]);
    }
    flip_clipPolygon = [];
    for (let c of clipPolygon) {
        flip_clipPolygon.push([c[0], c[1]]);
    }
            
    var cp1, cp2, s, e;
    var inside = function (p) {
        return (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0]);
    };
    var intersection = function () {
        var dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ],
            dp = [ s[0] - e[0], s[1] - e[1] ],
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0],
            n2 = s[0] * e[1] - s[1] * e[0], 
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0]);
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3];
    };
    var outputList = flip_subjectPolygon;
    cp1 = flip_clipPolygon[flip_clipPolygon.length-1];
    for (var j in flip_clipPolygon) {
        cp2 = flip_clipPolygon[j];
        var inputList = outputList;
        outputList = [];
        s = inputList[inputList.length - 1]; //last on the input list
        for (var i in inputList) {
            e = inputList[i];
            if (inside(e)) {
                if (!inside(s)) {
                    outputList.push(intersection());
                }
                outputList.push(e);
            }
            else if (inside(s)) {
                outputList.push(intersection());
            }
            s = e;
        }
        cp1 = cp2;
    }
    return outputList
}



/*
    x, y coordinates
*/
function ccw(A, B, C) {
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0]);
}
/*
    Return true if line segments AB and CD intersect
*/
function intersect(A,B,C,D) {
    return ccw(A,C,D) != ccw(B,C,D) && ccw(A,B,C) != ccw(A,B,D);
}

function polygon_is_self_intersecting(poly) {
    for (let i = 0; i < poly.length; i++) {
        let A = [poly[i][1], poly[i][0]];
        let B = [poly[(i+1)%poly.length][1], poly[(i+1)%poly.length][0]];
        for (let j = i+1; j < poly.length; j++) {
            let C = [poly[j][1], poly[j][0]];
            let D = [poly[(j+1)%poly.length][1], poly[(j+1)%poly.length][0]];

            if ((arraysEqual(A, C) || arraysEqual(A, D)) || (arraysEqual(B, C) || arraysEqual(B, D))) {
                continue;
            }

            if(intersect(A,B,C,D)) {
                return true;
            }
        }
    }
    return false;
}



/*
    https://stackoverflow.com/questions/45660743/sort-points-in-counter-clockwise-in-javascript

    takes y, x points
*/
function sort_clockwise(points) {


    let working_points = [];
    for (let point of points) {
        working_points.push({x: point[1], y: point[0]});
    }

    // Get the center (mean value) using reduce
    let center = working_points.reduce((acc, { x, y }) => {
        acc.x += x / working_points.length;
        acc.y += y / working_points.length;
        return acc;
    }, { x: 0, y: 0 });
    
    // Add an angle property to each point using tan(angle) = y/x
    let angles = working_points.map(({ x, y }) => {
        return { x, y, angle: Math.atan2(y - center.y, x - center.x) * 180 / Math.PI };
    });
    
    // Sort your points by angle
    let pointsSorted = angles.sort((a, b) => a.angle - b.angle);
    

    ret_points = [];
    for (let point of pointsSorted) {
        ret_points.push([point.y, point.x]);
    }
    return ret_points;
}


function sort_anti_clockwise(points) {
    let cw_pts = sort_clockwise(points);

    // then reverse the order
    let ccwPoints = cw_pts.reverse();

    // move the last point back to the start
    ccwPoints.unshift(ccwPoints.pop());

    //return ccwPoints;
    return ccwPoints;
}