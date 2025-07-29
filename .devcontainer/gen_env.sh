#!/bin/bash

set -e

BASE_IMAGE="ros:humble"
ENV_FILE=".devcontainer/.env"

echo "Generating $ENV_FILE ..."
echo "USER=$(whoami)" > "$ENV_FILE"
echo "USER_UID=$(id -u)" >> "$ENV_FILE"
echo "USER_GID=$(id -g)" >> "$ENV_FILE"
echo "BASE_IMAGE=$BASE_IMAGE" >> "$ENV_FILE"

echo "Done. Contents:"
cat "$ENV_FILE"
