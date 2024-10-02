#!/bin/bash

cp -r ./off-target/targetscan ./off-target/tmp
cp "$1" "$2" "$3" ./off-target/tmp/
cd ./off-target/tmp
mir=$(basename $1)
utr=$(basename $2)
orf=$(basename $3)

awk 'BEGIN{OFS="\t"} {x=$0; getline; print x,"9606",x,$1;}' $mir | sed 's/>//g' > sirnas_for_context_scores.txt
awk 'BEGIN{OFS="\t"} {x=$0; getline; print x,substr($1,2,7),"9606";}' $mir | sed 's/>//g' > sirnas.txt

awk 'BEGIN{OFS="\t"} {x=$0; getline; print x,"9606",$1;}' $utr | sed 's/>//g' > UTR.txt

awk 'BEGIN{OFS="\t"} {x=$0; getline; print x,"9606",$1;}' $orf | sed 's/>//g' > ORF.txt
awk 'BEGIN{OFS="\t"} {x=$0; getline; print x,"9606",length($1);}' $orf | sed 's/>//g' > ORF.length.txt

perl targetscan_70.pl sirnas.txt UTR.txt targetscan_70_output.txt

perl targetscan_70_BL_bins.pl UTR.txt > UTRs_median_BLs_bins.txt

perl targetscan_70_BL_PCT.pl sirnas.txt targetscan_70_output.txt UTRs_median_BLs_bins.txt > targetscan_70_output.BL_PCT.txt

perl targetscan_count_8mers.pl sirnas.txt ORF.txt > ORF_8mer_counts.txt

perl targetscan_70_context_scores.pl sirnas_for_context_scores.txt UTR.txt targetscan_70_output.BL_PCT.txt ORF.length.txt ORF_8mer_counts.txt Targets.BL_PCT.context_scores.txt

sed '1d' Targets.BL_PCT.context_scores.txt | awk 'BEGIN{FS="\t";OFS="\t"} {print $1,$3,$4,$28;}' | sort -k 1,2 > temp.txt
grep "6mer" temp.txt | awk 'BEGIN{OFS="\t"} {print $1,$2,$4}' > 6mer.txt
grep "7mer-1a" temp.txt | awk 'BEGIN{OFS="\t"} { if( $28< -0.01 ) {print $1,$2,$4;} }' > 7merA1.txt
grep "7mer-m8" temp.txt | awk 'BEGIN{OFS="\t"} { if( $28< -0.02 ) {print $1,$2,$4;} }' > 7merm8.txt
grep "8mer-1a" temp.txt | awk 'BEGIN{OFS="\t"} { if( $28< -0.03 ) {print $1,$2,$4;} }' > 8mer.txt
cat 6mer.txt 7merA1.txt 7merm8.txt 8mer.txt | sort -k 1,2 > temp.txt

awk 'BEGIN{OFS="\t";target="";sirna="";totscore=0.0;} {
        if( target==$1 && sirna==$2 ) { totscore+=$3; }
        else { print target,sirna,-totscore; target=$1; sirna=$2; totscore=$3; }
        } END{print target,sirna,-totscore;}' temp.txt | sed '1d' >> offtarget_predictions.tab

cd ../../
mv ./off-target/tmp/offtarget_predictions.tab ./data/infer/"$4"/targetscan.tab
rm -rf ./off-target/tmp
