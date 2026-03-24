
<h2 align="center">
Knowledge Graph Pruning for Recommendation
</h2>

This is the Pytorch implementation for KGPR.

## Dependencies
- pytorch==1.11.0
- numpy==1.21.5
- scipy==1.7.3
- torch-scatter==2.0.9
- scikit-learn==0.24.2

## Download the dataset
You can find the dataset in this [link](https://rec.ustc.edu.cn/share/aebef1e0-5a6a-11f0-b90c-51ece35236d1), extract code is `ml73`.

then move this dataset to `./data`
## training

### Pruned model training

#### amazon-book
```
python main.py --dataset amazon-book 
```

#### last-fm
```
python main.py --dataset last-fm 
```

#### alibaba-fashion
```
python main.py --dataset alibaba-fashion 
```

### Create pruned knowledge graph
Normally, the kg can be created automanticly. If your code is shut down unexpectly, you can generate the kg file via following command.
```
python main.py --pretrain_model_path=<saved model path>
```

Then you can find the pruned kg in your `saved_kg`

## Training with pruned knowledge graph
Now, we include the KGIN code in our repo.
So you can immediately run the code.
- move the pruned kg file to the corresponding data dir. For example, we can move `amazon-book_kgpr_95.txt` to `data/amazon-book`
- change the workspace into KGIN, i.e., `cd KGIN`
- run the base model via `run.sh` or we can run the script
```python
python main.py --dataset amazon-book --kg_file amazon-book_kgpr_95 --batch_size 4096
```
