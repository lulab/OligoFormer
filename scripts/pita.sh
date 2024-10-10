#!/bin/bash

cp "$1" "$2" "$3" ./off-target/pita/
cd ./off-target/pita
utr=$(basename $1)
mir=$(basename $2)
orf=$(basename $3)
perl pita_prediction.pl -utr "$utr" -mir "$mir" -upstream "$orf" -prefix "$4" 
rm -rf "$utr" "$mir" "$orf"
cd ../..
mv ./off-target/pita/"$4"_pita_results_targets.tab ./data/infer/"$4"/pita.tab
