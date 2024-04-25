let password_visible = false;

$(document).ready(function(){

    $("#error_message").hide();
    $("#sign_in_loader").hide();

    $("form").submit(function(e) {
        e.preventDefault();

        let provided_username = $("#username").val();
        let provided_password = $("#password").val();

        $("#error_message").hide();
        $("#sign_in_loader").show();
        disable_green_buttons(["sign_in_button"]);
        $("#sign_in_button_text").css("cursor", "default");

        $.post($(location).attr('href'),
        {
            username: provided_username,
            password: provided_password
        },
        
        function(response, status) {

            if (response.error || response.not_found) {
                $("#sign_in_loader").hide();
                enable_green_buttons(["sign_in_button"]);
                $("#sign_in_button_text").css("cursor", "pointer");

                if (response.error) {
                    $("#error_message").html("Sorry, an error occurred during the sign-in.");
                }
                else {
                    $("#error_message").html("Your username/password combination is incorrect.");
                }
                $("#error_message").show();
            }
            else {
                window.location.href = response.redirect;
            }
        });
    });


    $("#password_vis_toggle").click(function() {

        if ($("#password").attr("type") === "password") {
            $("#password").attr("type", "text");
            $("#password_vis_toggle").removeClass("fa-solid fa-eye-slash")
            $("#password_vis_toggle").addClass("fa-solid fa-eye");
        }
        else {
            $("#password").attr("type", "password");
            $("#password_vis_toggle").removeClass("fa-solid fa-eye")
            $("#password_vis_toggle").addClass("fa-solid fa-eye-slash");
        }

    });



});