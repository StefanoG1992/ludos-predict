#!/bin/sh

# shellcheck disable=SC2039
source .env

cd "${WORKDIR}" || exit

# shellcheck disable=SC2046
printf "\nPlotting 2D graph\n"
pipenv run python src/main.py plot -p 2d -i "${DATAPATH}" -s "${OUTPUTDIR}"
printf "\nPlotting shapley coefficients\n"
pipenv run python src/main.py plot -p shapley -i "${DATAPATH}" -s "${OUTPUTDIR}"
