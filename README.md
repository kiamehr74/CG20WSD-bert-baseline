This is the BERT baseline model for CG20WSD dataset

## Requirements

- CUDA version 10.2 or higher
- torch version 1.4.0
```bash
pip install torch==1.4.0
```
- HuggingFace transformers package version 2.7.0 or higher
```bash
pip install transformers==2.7.0
```


## Usage
To run the script,  use the following command 
```bash
python run.py -model bert-base-uncased -config config/default.json -save_model 
```

Parameters:
- `-model` : Indicates the model, currently supporting `bert-base-uncased` and `bert-large-uncased`
- `-config` : Path to the config file
- `-save_model` : put this switch if you want to save the fine-tuned models.
