
let usernames = [];
let object_names = [];



function add_objects() {

    $("#class_list").empty();

    for (let object_name of object_names) {

        $("#class_list").append(
            `<tr style="border-bottom: 1px solid #4c6645; height: 40px"">` + 
                `<td style="width: 50%"></td>` +
                `<td>` +
                    `<div style="width: 200px" class="object_entry">${object_name}</div>` +
                `</td>` +
                `<td style="width: 50%"></td>` +
            `</tr>` 
        );
    }

    let first_row = document.getElementById("class_list").rows[0]
    if (first_row != null) {
        first_row.scrollIntoView();
    }
}

function add_users() {

    $("#user_list").empty();

    for (let username of usernames) {

        $("#user_list").append(
            `<tr style="border-bottom: 1px solid #4c6645; height: 40px"">` + 
                `<td style="width: 50%"></td>` +
                `<td>` +
                    `<div style="width: 200px" class="object_entry">${username}</div>` +
                `</td>` +
                `<td style="width: 50%"></td>` +
                `<td>` +
                    `<button class="button-green button-green-hover" style="margin-top: 2px; width: 25px; height: 25px; border-radius: 5px; font-size: 12px;" type="button" onclick="update_password_modal('${username}')">` +
                        `<i class="fa-solid fa-gear"></i>` +
                    `</button>` +
                `</td>` +
                `<td>` +
                    `<div style="width: 10px"></div>` +
                `</td>` +
            `</tr>` 
        );
    }

    let first_row = document.getElementById("user_list").rows[0]
    if (first_row != null) {
        first_row.scrollIntoView();
    }
}



function update_password_modal(username) {
    show_modal_message(
        "Update User Password",
        `<table>` +
            `<tr>` +
                `<td>` + 
                    `<div class="table_head" style="width: 120px; padding-right: 10px">Username</div>` +
                `</td>` +
                `<td>` +
                    `<div style="width: 330px">` +
                        username +
                    `</div>` +
                `</td>` +
            `</tr>` +
            `<tr style="height: 5px">` +
            `<tr>` +
                `<td>` + 
                    `<div class="table_head" style="width: 120px; padding-right: 10px">Password</div>` +
                `</td>` +
                `<td>` +
                    `<div style="width: 330px">` +
                        `<input id="password_input" class="nonfixed_input" style="width: 100%">` +
                    `</div>` +
                `</td>` +
            `</tr>` +
        `</table>` +

        `<div style="height: 30px"></div>` +
        `<div style="text-align: center">` +
            `<button onclick="update_user_password('${username}')" class="button-green button-green-hover" `+
                    `style="width: 200px">Update Password` +
            `</button>` +
        `</div>`
    );

}
function update_user_password(username) {

    let password = $("#password_input").val();

    $.post($(location).attr('href'),
    {
        action: "update_user_password",
        username: username,
        password: password
    },
    
    function(response, status) {

        if (response.error) {  
            show_modal_message("Error", response.message);  
        }
        else {
            show_modal_message("Success", "The user password has been successfully updated.");
        }
    });

}




function add_object_class() {

    let object_name = $("#object_name_input").val();

    $.post($(location).attr('href'),
    {
        action: "add_object_class",
        object_name: object_name
    },
    
    function(response, status) {

        if (response.error) {  
            show_modal_message("Error", response.message);  
        }
        else {
            show_modal_message("Success", "The object class has been successfully added.");

            object_names.push(object_name);
            natsort(object_names);

            add_objects();
        }
    });
}




function create_user_account() {

    let username = $("#username_input").val();
    let password = $("#password_input").val();

    $.post($(location).attr('href'),
    {
        action: "create_user_account",
        username: username,
        password: password
    },
    
    function(response, status) {

        if (response.error) {  
            show_modal_message("Error", response.message);  
        }
        else {
            show_modal_message("Success", "The user account has been successfully created.");

            usernames.push(username);
            natsort(usernames);

            add_users();
        }
    });
}


$(document).ready(function() {

    $("#site_home_button").attr("href", get_AC_PATH() + "/admin");


    usernames = [];
    for (let user of data["users"]) {
        usernames.push(user["username"]);
    }
    natsort(usernames);

    object_names = data["object_names"];
    natsort(object_names);


    add_users();
    add_objects();




    $("#add_cls_button").click(function() {
        show_modal_message(
            "Add Object Class",
            `<table>` +
                `<tr>` +
                    `<td>` + 
                        `<div class="table_head" style="width: 120px; padding-right: 10px">Object Name</div>` +
                    `</td>` +
                    `<td>` +
                        `<div style="width: 330px">` +
                            `<input id="object_name_input" class="nonfixed_input" style="width: 100%">` +
                        `</div>` +
                    `</td>` +
                `</tr>` +
                `<tr style="height: 5px">` +
            `</table>` +

            `<div style="height: 30px"></div>` +
            `<div style="text-align: center">` +
                `<button onclick="add_object_class()" class="button-green button-green-hover" `+
                        `style="width: 200px">Add Object Class` +
                `</button>` +
            `</div>`
        );
    });



    $("#add_usr_button").click(function() {
        show_modal_message(
            "Create User Account",
            `<table>` +
                `<tr>` +
                    `<td>` + 
                        `<div class="table_head" style="width: 120px; padding-right: 10px">Username</div>` +
                    `</td>` +
                    `<td>` +
                        `<div style="width: 330px">` +
                            `<input id="username_input" class="nonfixed_input" style="width: 100%">` +
                        `</div>` +
                    `</td>` +
                `</tr>` +
                `<tr style="height: 5px">` +
                `<tr>` +
                    `<td>` + 
                        `<div class="table_head" style="width: 120px; padding-right: 10px">Password</div>` +
                    `</td>` +
                    `<td>` +
                        `<div style="width: 330px">` +
                            `<input id="password_input" class="nonfixed_input" style="width: 100%">` +
                        `</div>` +
                    `</td>` +
                `</tr>` +
            `</table>` +

            `<div style="height: 30px"></div>` +
            `<div style="text-align: center">` +
                `<button onclick="create_user_account()" class="button-green button-green-hover" `+
                        `style="width: 200px">Create User Account` +
                `</button>` +
            `</div>`
        );
    });
});