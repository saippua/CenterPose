## This file contains all trains. Each training results in an approx 800mb model.
## total ~22.4 GB space requirement for all models. This can be reduced by removing 
## checkpoints and/or log data

# # Vanilla centerpose
python ./train_final.py -d 1 --displacement --multi_object --scale --prefix rep1
python ./train_final.py -d 2 --displacement --multi_object --scale --prefix rep1
python ./train_final.py -d 3 --displacement --multi_object --scale --prefix rep1
python ./train_final.py -d 4 --displacement --multi_object --scale --prefix rep1

# # Scale modification only
python ./train_final.py -d 1 --displacement --multi_object --prefix rep1
python ./train_final.py -d 2 --displacement --multi_object --prefix rep1
python ./train_final.py -d 3 --displacement --multi_object --prefix rep1
python ./train_final.py -d 4 --displacement --multi_object --prefix rep1

# # Scale and Displacement modification
python ./train_final.py -d 1 --multi_object --prefix rep1
python ./train_final.py -d 2 --multi_object --prefix rep1
python ./train_final.py -d 3 --multi_object --prefix rep1
python ./train_final.py -d 4 --multi_object --prefix rep1

# # # All modifications
# python ./train_final.py -d 1 
python ./train_final.py -d 2 --prefix rep1
python ./train_final.py -d 3 --prefix rep1
python ./train_final.py -d 4 --prefix rep1

# # Vanilla centerpose
python ./train_final.py -d 1 --displacement --multi_object --scale --prefix rep2
python ./train_final.py -d 2 --displacement --multi_object --scale --prefix rep2
python ./train_final.py -d 3 --displacement --multi_object --scale --prefix rep2
python ./train_final.py -d 4 --displacement --multi_object --scale --prefix rep2

# # Scale modification only
python ./train_final.py -d 1 --displacement --multi_object --prefix rep2
python ./train_final.py -d 2 --displacement --multi_object --prefix rep2
python ./train_final.py -d 3 --displacement --multi_object --prefix rep2
python ./train_final.py -d 4 --displacement --multi_object --prefix rep2

# # Scale and Displacement modification
python ./train_final.py -d 1 --multi_object --prefix rep2
python ./train_final.py -d 2 --multi_object --prefix rep2
python ./train_final.py -d 3 --multi_object --prefix rep2
python ./train_final.py -d 4 --multi_object --prefix rep2

# # # All modifications
# python ./train_final.py -d 1 --prefix rep2
python ./train_final.py -d 2 --prefix rep2
python ./train_final.py -d 3 --prefix rep2
python ./train_final.py -d 4 --prefix rep2

# # Vanilla centerpose
python ./train_final.py -d 1 --displacement --multi_object --scale --prefix rep3
python ./train_final.py -d 2 --displacement --multi_object --scale --prefix rep3
python ./train_final.py -d 3 --displacement --multi_object --scale --prefix rep3
python ./train_final.py -d 4 --displacement --multi_object --scale --prefix rep3

# # Scale modification only
python ./train_final.py -d 1 --displacement --multi_object --prefix rep3
python ./train_final.py -d 2 --displacement --multi_object --prefix rep3
python ./train_final.py -d 3 --displacement --multi_object --prefix rep3
python ./train_final.py -d 4 --displacement --multi_object --prefix rep3

# # Scale and Displacement modification
python ./train_final.py -d 1 --multi_object --prefix rep3
python ./train_final.py -d 2 --multi_object --prefix rep3
python ./train_final.py -d 3 --multi_object --prefix rep3
python ./train_final.py -d 4 --multi_object --prefix rep3

# # # All modifications
# python ./train_final.py -d 1 --prefix rep3
python ./train_final.py -d 2 --prefix rep3
python ./train_final.py -d 3 --prefix rep3
python ./train_final.py -d 4 --prefix rep3

# # Vanilla centerpose
python ./train_final.py -d 1 --displacement --multi_object --scale --prefix rep4
python ./train_final.py -d 2 --displacement --multi_object --scale --prefix rep4
python ./train_final.py -d 3 --displacement --multi_object --scale --prefix rep4
python ./train_final.py -d 4 --displacement --multi_object --scale --prefix rep4

# # Scale modification only
python ./train_final.py -d 1 --displacement --multi_object --prefix rep4
python ./train_final.py -d 2 --displacement --multi_object --prefix rep4
python ./train_final.py -d 3 --displacement --multi_object --prefix rep4
python ./train_final.py -d 4 --displacement --multi_object --prefix rep4

# # Scale and Displacement modification
python ./train_final.py -d 1 --multi_object --prefix rep4
python ./train_final.py -d 2 --multi_object --prefix rep4
python ./train_final.py -d 3 --multi_object --prefix rep4
python ./train_final.py -d 4 --multi_object --prefix rep4

# # # All modifications
python ./train_final.py -d 1 --prefix rep4
python ./train_final.py -d 2 --prefix rep4
python ./train_final.py -d 3 --prefix rep4
python ./train_final.py -d 4 --prefix rep4

# # Vanilla centerpose
python ./train_final.py -d 1 --displacement --multi_object --scale --prefix rep5
python ./train_final.py -d 2 --displacement --multi_object --scale --prefix rep5
python ./train_final.py -d 3 --displacement --multi_object --scale --prefix rep5
python ./train_final.py -d 4 --displacement --multi_object --scale --prefix rep5

# # Scale modification only
python ./train_final.py -d 1 --displacement --multi_object --prefix rep5
python ./train_final.py -d 2 --displacement --multi_object --prefix rep5
python ./train_final.py -d 3 --displacement --multi_object --prefix rep5
python ./train_final.py -d 4 --displacement --multi_object --prefix rep5

# # Scale and Displacement modification
python ./train_final.py -d 1 --multi_object --prefix rep5
python ./train_final.py -d 2 --multi_object --prefix rep5
python ./train_final.py -d 3 --multi_object --prefix rep5
python ./train_final.py -d 4 --multi_object --prefix rep5

# # # All modifications
python ./train_final.py -d 1 --prefix rep5
python ./train_final.py -d 2 --prefix rep5
python ./train_final.py -d 3 --prefix rep5
python ./train_final.py -d 4 --prefix rep5
