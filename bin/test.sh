#!/bin/sh

cd "${WORKDIR}" || exit

# shellcheck disable=SC2039
source .env

# if 3rd parameter is passed, it will be treated as scoring
if [ -z "$3" ]; then
  metric="f1_score"
else
  metric="$3"
fi

# logs are saved as _timestamp - removing all logs so only last one is present
m -r "${WORKDIR}"/logs/ml_predict_*.log

# shellcheck disable=SC2039
if [[ "$2" == "-O" ]]; then
  echo "Enter optimization"
  echo "Metric used: $metric"
  pipenv run python src/main.py -i "${DATAPATH}" -s "${OUTPUTDIR}" -l "${WORKDIR}"/logs test -m "$1" -O -s "$metric"
else
  echo "Do not enter optimization"
  echo "Metric used: $metric"
  pipenv run python src/main.py -i "${DATAPATH}" -s "${OUTPUTDIR}" -l "${WORKDIR}"/logs test -m "$1" -s "$metric"
fi
