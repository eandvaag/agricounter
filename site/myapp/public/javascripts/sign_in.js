let password_visible = false;


$(document).ready(function(){

    $("#error_message").hide();

    $('form').submit(function(e) {
        e.preventDefault();

        $.post($(location).attr('href'),
        {
            username: $("#username").val(),
            password: $("#password").val()
        },
        
        function(response, status) {

            if (response.error) {
                $("#error_message").html("Sorry, an error occurred during the sign-in.");
                $("#error_message").show();
            }
            else if (response.not_found) {
                $("#error_message").html("Your username/password combination is incorrect.");
                $("#error_message").show();
            }
            else if (response.maintenance) {
                $("#error_message").html("Sorry, the site is currently under maintenance.");
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