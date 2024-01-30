'use strict';

var bcrypt = require('bcrypt');


module.exports = (sequelize, DataTypes) => {
    var users = sequelize.define('users', {
        username: {
            type: DataTypes.STRING,
            allowNull: false,
            primaryKey: true
        },
        password: {
            type: DataTypes.STRING,
            allowNull: false
        },
        is_admin: {
            allowNull: false,
            type: DataTypes.BOOLEAN
        }
    });

    users.addHook('beforeCreate', (user) => {
        const salt = bcrypt.genSaltSync();
        user.password = bcrypt.hashSync(user.password, salt);
    });

    users.prototype.check_password = function(password) {
        return bcrypt.compareSync(password, this.password);
    }

    return users;
}

