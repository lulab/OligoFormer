# nohup python -u script/main.py --n_head 4 --n_layers 1 > output41 2>&1 &
# nohup python -u script/main.py --n_head 4 --n_layers 2 > output42 2>&1 &
# nohup python -u script/main.py --n_head 4 --n_layers 3 > output43 2>&1 &
# nohup python -u script/main.py --n_head 8 --n_layers 1 > output81 2>&1 &
# nohup python -u script/main.py --n_head 8 --n_layers 2 > output82 2>&1 &
# nohup python -u script/main.py --n_head 8 --n_layers 3 > output83 2>&1 &
# nohup python -u script/main.py --n_head 16 --n_layers 1 > output161 2>&1 &
# nohup python -u script/main.py --n_head 16 --n_layers 2 > output162 2>&1 &
# nohup python -u script/main.py --n_head 16 --n_layers 3 > output163 2>&1 &

for head in 4 8 16
do
for layer in 1 2 3
do
nohup python -u script/main.py --n_head $head --n_layers $layer > output$head$layer 2>&1 &
done
done


nohup python -u script/main.py  > outputHu 2>&1 &
nohup python -u script/main.py --kfold 10 > outputHu_10 2>&1 &

nohup python -u script/main.py  > outputnew 2>&1 &
nohup python -u script/main.py --kfold 10 > outputnew_10 2>&1 &

nohup python -u script/main.py  > outputTaka 2>&1 &
nohup python -u script/main.py --kfold 10 > outputTaka_10 2>&1 &
