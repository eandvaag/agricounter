#!/bin/sh -e
# wait-for-postgres.sh
# Adapted from https://docs.docker.com/compose/startup-order/

# Expects the necessary PG* variables.

echo 'Waiting for postgres'



until PGPASSWORD=$DB_PASSWORD psql -c '\l' -h db -p 5432 -U $DB_USER $DB_SCHEMA; do
  echo >&2 "$(date +%Y%m%dt%H%M%S) Postgres is unavailable - sleeping"
  sleep 1
done
echo >&2 "$(date +%Y%m%dt%H%M%S) Postgres is up - executing command"



USERS_TABLE_EXISTS=`PGPASSWORD=$DB_PASSWORD psql -c "SELECT EXISTS ( SELECT 1 FROM pg_catalog.pg_class WHERE relname='users' AND relkind='r')" -h db -p 5432 -U $DB_USER $DB_SCHEMA | head -3 | tail -1`

if [ $USERS_TABLE_EXISTS == 't' ]; then
  echo "Users table exists, not running migrations and seeders"
else
  echo 'Running db:migrate'
  npx sequelize-cli db:migrate

  echo 'Running db:seed:all'
  npx sequelize-cli db:seed:all

fi


echo 'Starting npm'
npm run start
