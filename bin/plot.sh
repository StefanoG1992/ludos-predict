#!/bin/sh

# shellcheck disable=SC2039
source .env

cd "${WORKDIR}" || exit

# shellcheck disable=SC2046
printf "\nPlotting 2D graph\n"
pipenv run python src/main.py -i "${DATAPATH}" -s "${OUTPUTDIR}" plot -p 2d
printf "\nPlotting shapley coefficients\n"
pipenv run python src/main.py -i "${DATAPATH}" -s "${OUTPUTDIR}" plot -p shapley
