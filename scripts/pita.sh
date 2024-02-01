#!/bin/bash

cd ./PITA || exit 1
perl pita_prediction.pl -utr "$1" -mir "$2" -upstream "$3" -prefix "$4" 