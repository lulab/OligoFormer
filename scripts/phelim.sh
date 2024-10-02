#!/bin/bash

cp -r ./PheLiM ./tmp
cd ./tmp/PheLiM || exit 1
bash predict-off-targets.sh "$1" "$2" "$3" "$4"
cd ../../
