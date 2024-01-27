<div align="center">
  
# ~ _tinyllama_ ~

<img src="https://github.com/miftahmoha/tinyllama/assets/102898329/43f42dfc-6b6c-4865-bdde-952785674fde" alt="TinyLlama Logo" width=550></img>


_Model classes and pre-training utilities for a tiny version of Llama in PyTorch._

</div>

## Installation 🚀

``` bash
pip install tinyllama
```

## Pre-training a model 🏋‍♀

### Initializing a tokenizer

With a simple character-level tokenizer:

```python
from tinyllama.tokenizers import CharacterTokenizer
tokenizer = CharacterTokenizer()
# '|' is the default eos_token
tokenizer.add_eos_tokens()
```

To turn a corpus into tokens:

```python
tokens  = tokenizer.tokenize(corpus)
```

### Initializing a Llama model

```python
from tinyllama import Llama
model = Llama(context_window=500, emb_dim=10, n_heads=2, n_blocks=2)
```

#### Multi-Query attention

```python
model = Llama(context_window=500, emb_dim=10, n_heads=2, n_blocks=2, gq_ratio=2)
```

The parameter gq_ratio represents the ratio $\frac{number \ of \ heads}{number \  of \ queries/keys}$, it is set to 1 by default.

The configuration above builds a Llama model with the number of heads being twice as much as the number of queries/keys.

### Launching a pre-training job

```python
from tinyllama import TrainConfig, Trainer
TrainConfig = TrainConfig(batch_size=32, epochs=50, log_interval=15)
Trainer = Trainer(TrainConfig)
Trainer.run(model, tokens)
```

## Diagnosis 😷

Diagnosis class run a training job on a copy of the model and returns training information that could be useful to the user.

### Diagnosing the learning rate

Returns a plot representing the loss for each learning rate, _the scale for the argument start and end is logarithmic_.

```python
from tinyllama.diagnosis import LrDiagnose                                                                                                                                                                                                       LrDiagnose = LrDiagnose(start=-5, end=0, n_lrs=50)                                                                   # LrDiagnose.run(model, tokens, TrainConfig)
LrDiagnose = LrDiagnose(start=-5, end=0, n_lrs=50)
LrDiagnose.run(model, tokens, TrainConfig)
```

### Diagnosing the gradients

Returns a histogram representing the distribution of the gradients, doesn't run additional training jobs.

```python
from tinyllama.diagnosis import GradDiagnose
GradDiagnose = GradDiagnose(num_params_to_track=1500)
GradDiagnose.run(model)
```

### Diagnosing the activation layers (SwiGLU layers)

Returns a histogram representing the distribution of the activation layers.

```python
from tinyllama.diagnosis import SwigluDiagnose
SwigluDiagnose = SwigluDiagnose(num_embeddings_for_histogram=50, track_direction="forward" )
SwigluDiagnose.run(model, tokens, TrainConfig)
```

### Diagnosing the gradients/data ratios

Returns a plot representing the gradient/data ratio in each step of the training.

```python
from tinyllama.diagnosis import SwigluDiagnose
GdrDiagnose = GdrDiagnose(num_params_to_track=5, num_iters=150)
GdrDiagnose.run(model, tokens, TrainConfig)
```

### Hyperparameter tuning with GPTune ⚙️

GPTune facilitates hyperparameter tuning by leveraging Gaussian Processes as a means to optimize the tuning process.

```python
from tinyllama.gptuner import GPTuneConfig, GPTune
GPTuneConfig = GPTuneConfig(num_training_samples=100, hyperparams_to_tune=["epochs", "n_heads"], l_bounds=[10, 2], u_bounds=[50, 5], num_evaluations=500)
GPTune = GPTune(GPTuneConfig)
GPTune.run(model, tokens, TrainConfig)
```

## Generating ✍

Generates a response to a prompt.

```python
from tinyllama import generate
# kv_cache is set to True by default.
generate(model, prompt, max_tokens=900, kv_cache=True)
```

## Parsing 📜

Parses single or multiple files.

```python
# ".txt" files
from tinyllama.readers import get_text
corpus = get_text("./txt_path")

# ".pdf" files
from tinyllama.readers import get_pdf_text
corpus = get_pdf_text("./pdf_path")
```

To parse multiple files:

```python
# ".txt" files
from tinyllama.readers import get_text
corpus = ''.join(get_text(pdf_path) for txt_path in txt_paths)

# ".pdf" files
from tinyllama.readers import get_pdf_text
corpus = ''.join(get_pdf_text(pdf_path) for pdf_path in pdf_paths)
```
