### Multi-Task Learning
-------

## Introduction

This is an example of how to construct a multi-task neural net in Tensorflow. Here we're looking at Natural Language Processing ("NLP"), specifically on whether by learning about Part of Speech (POS), Shallow Parsing (Chunking) and Name Entity Recognition at the same time we can improve performance on all three.

## Network Structure

Our network looks a little bit like this, with Task 1 being Part of Speech (POS) and Task 2 being Chunk(Task three NER not shown in the picture but it is built on the top of POS layer similar to chunking):

<img src='https://jg8610.github.io/images/joint_op.png'>

As you can see, you can train either tasks separately (by calling the individual training ops), or you can train the tasks jointly (by calling the join training op).

We have also added in an explicit connection from POS to Chunk and POS to NER which actually makes the network into something similar to a ladder network with an explicit hidden state representation.

## Quick Start (Linux)

* You have to install following packages:   
python3  
numpy  
pandas  
tensorflow(I ran the model on (1.14.0)  


## Training

### POS Single
```bash
python3 multi_task.py --model_type "POS" \
       	             --dataset_path "./data" \
		                 --save_path "./data/outputs"

```
### Chunk Single
```bash
python3 multi_task.py --model_type "CHUNK" \
       	             --dataset_path "./data" \
		                 --save_path "./data/outputs"
```
### NER Single
```bash
python3 multi_task.py --model_type "NER" \
       	             --dataset_path "./data" \
		                 --save_path "./data/outputs"
```

### Joint
```bash
python3 multi_task.py --model_type "JOINT" \
       	             --dataset_path "./data" \
		                 --save_path "./data/outputs"

```

## Output:
* You can then print out the evaluations by typing:
``python generate_results.py --path "./data/outputs/predictions/"``


Note: This model has been forked from Jonathan Godwin's multi-tasking model which is for Chunking and POS. I have added an extra NER task to the model. I have edited the readme according to the changes which I have brought in.
