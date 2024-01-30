function show_modal_message(head, message, modal_width=650, display=true) {

    $("#modal_head").empty();
    $("#modal_body").empty();

    $("#modal_head").append(
    `<span class="close close-hover" id="modal_close">&times;</span>` +
    `<p id="modal_head_text">` + head + `</p>`);

    $("#modal_body").append(`<p id="modal_message" style="overflow-wrap: break-word" align="left"></p>`);
    
    
    $("#modal_message").html(message);
    
    $("#modal_close").click(function() {
        close_modal();
    });

    $(".modal-content").width(modal_width);

    if (display) {
        $("#modal").css("display", "block");
    }

    
}

function close_modal() {
    $("#modal").css("display", "none");
}

