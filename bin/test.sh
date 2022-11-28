#!/bin/sh

cd "${WORKDIR}" || exit

# shellcheck disable=SC2039
source .env

# if 3rd parameter is passed, it will be treated as scoring
if [ -z "$2" ]; then
  metric="f1_score"
else
  metric="$2"
fi

# logs are saved as _timestamp - removing all logs so only last one is present
rm -r "${WORKDIR}"/logs/ml_predict_*.log

# shellcheck disable=SC2039
if [[ "$3" == "-O" ]]; then
  echo "Optimization performed"
  echo "Metric used: $metric"
  pipenv run python src/main.py -i "${DATAPATH}" -s "${OUTPUTDIR}" -l "${WORKDIR}"/logs test -m "$1" -O -s "$metric"
else
  echo "No optimization step"
  echo "Metric used: $metric"
  pipenv run python src/main.py -i "${DATAPATH}" -s "${OUTPUTDIR}" -l "${WORKDIR}"/logs test -m "$1" -s "$metric"
fi
