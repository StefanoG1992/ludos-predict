#!/bin/sh

cd "${WORKDIR}" || exit

# shellcheck disable=SC2039
source .env

# shellcheck disable=SC2039
if [[ "$2" == "-O" ]]; then
  echo "Enter optimization"
  pipenv run python src/main.py -i "${DATAPATH}" -s "${OUTPUTDIR}" -l "${WORKDIR}"/logs test -m "$1" -O
else
  echo "Do not enter optimization"
  pipenv run python src/main.py -i "${DATAPATH}" -s "${OUTPUTDIR}" -l "${WORKDIR}"/logs test -m "$1"
fi
