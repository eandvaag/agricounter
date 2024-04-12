



function get_global_metrics(region_type) {

    let data = {};
    let m_labels = [
        "True Positives (IoU=.50, conf>.50)",
        "False Positives (IoU=.50, conf>.50)",
        "False Negatives (IoU=.50, conf>.50)"
    ];
    let d_labels = [
        "Accuracy (IoU=.50, conf>.50)",
        "F1 Score (IoU=.50, conf>.50)",
        "Precision (IoU=.50, conf>.50)",
        "Recall (IoU=.50, conf>.50)"
    ]
    for (let object_class of metadata["object_classes"]) {
        let obj_metrics = metrics[object_class];
        
        data[object_class] = {};
        for (let m_label of m_labels) {
            data[object_class][m_label] = 0;
            for (let image_name of Object.keys(obj_metrics[m_label])) {
                for (let reg_val of obj_metrics[m_label][image_name][region_type]) {
                    data[object_class][m_label] += reg_val;
                }
            }
        }

        let tp = data[object_class]["True Positives (IoU=.50, conf>.50)"];
        let fp = data[object_class]["False Positives (IoU=.50, conf>.50)"];
        let fn = data[object_class]["False Negatives (IoU=.50, conf>.50)"];

        let prec, rec, acc, f1;
        if (tp == 0 && fp == 0 && fn == 0) {
            prec = 1.0;
            rec = 1.0;
            acc = 1.0;
            f1 = 1.0;
        }
        else if ((tp == 0 && fp > 0 && fn == 0) || (tp == 0 && fp == 0 && fn > 0)) {
            prec = 0.0;
            rec = 0.0;
            acc = 0.0;
            f1 = 0.0;
        }
        else {
            prec = tp / (tp + fp);
            rec = tp / (tp + fn);
            acc = tp / (tp + fp + fn);
            f1 = 2 * ((prec * rec) / (prec + rec));
        }

        data[object_class]["Precision (IoU=.50, conf>.50)"] = prec;
        data[object_class]["Recall (IoU=.50, conf>.50)"] = rec;
        data[object_class]["Accuracy (IoU=.50, conf>.50)"] = acc;
        data[object_class]["F1 Score (IoU=.50, conf>.50)"] = f1;
    }

    let all_labels = m_labels.concat(d_labels);
    data["Class Average"] = {};
    for (let label of all_labels) {
        data["Class Average"][label] = 0;
        for (let object_class of metadata["object_classes"]) {
            data["Class Average"][label] += data[object_class][label];
        }
        data["Class Average"][label] /= metadata["object_classes"].length;
    }
    
    return data;

}

function show_performance_modal() {


    // show_modal_message(


    // );


}

function print_class_accuracies(region_type) {

    let global_metrics = get_global_metrics(region_type);
    for (let class_name of Object.keys(global_metrics)) {
        let num = global_metrics[class_name]["Accuracy (IoU=.50, conf>.50)"] * 100;
        // let v = Math.round(num * 100) / 100;
        console.log(class_name.padStart(30), ":   ", num.toFixed(2));
    }

}

function draw_summary_chart() {

    




}