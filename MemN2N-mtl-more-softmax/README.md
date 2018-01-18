# End-to-end Memory Networks for Dialog
Modification of the implementation of Memory Networks from [Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/abs/1605.07683) in Tensorflow. Tested on the Personalized Dialogdataset. 

Adapted from [vyraun's implementation](https://github.com/vyraun/chatbot-MemN2N-tensorflow).

The purpose of this model is to provide a MTL model that will hopfully have a better generalization error.

## Usage

Train the model
```
python single_dialog.py --train True --task_id 1
```

Test the trained model
```
python single_dialog.py --train False --task_id 1
```

One can also re-compute an experiment using 

```
python single_diagog.py --experiment experiment_name
```

where `experiment_name` can be found in the code. But roughly,

- `test` will run a fast experiment to test the code
- `split-by-profile` train on data from all profiles and test on a profile basis 

**Remark** For `split-by-profile` the script `scripts/merge_split_by_profile.sh` has to be called once before (and from 
`script` directory).

## Requirements

See `../requirements.yml` (to be used in a conda environment).

The code is running on tensorflow 1.14 with python3.


