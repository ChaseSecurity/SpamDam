## modules
* senario1.py -- code for the attack in senario I 
* senario2.py -- code for the attack in senario II
* senario3.py -- code for the attack in senario III

## how to run
```
# --p : poisoning rate
# --train : path of training dataset in csv format 
# --test : path of testing dataset in csv format 
# --model : path of model to save
# --ham: the non spam dataset used to be injected as spam

python senario1.py --p 0.01 --ham /path/to/ham/dataset.csv --train /path/to/train.csv --test /path/to/test.csv --model /path/to/save/model

# --p : poisoning rate
# --keyword : the target keyword
# --train : path of training dataset in csv format 
# --test : path of testing dataset in csv format 
# --model : path of model to save
# --ham: the non spam dataset used to be injected as spam

python senario2.py --p 0.01 --keyword google --ham /path/to/ham/dataset.csv --train /path/to/train.csv --test /path/to/test.csv --model /path/to/save/model

# --p : poisoning rate
# --keyword : the target keyword
# --train : path of training dataset in csv format 
# --test : path of testing dataset in csv format 
# --model : path of model to save

python senario2.py --p 0.01 --keyword google --train /path/to/train.csv --test /path/to/test.csv --model /path/to/save/model
```