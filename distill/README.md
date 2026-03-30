# Distillation

`pip install torchtune torch torchvision torchao==0.7`
`pip install "lm_eval[hf]==0.4.7" transformers==4.48.3`

# Tutorial Distillation
# 1B

|    Tasks     |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------|------:|------|-----:|------|---|-----:|---|-----:|
|truthfulqa_mc2|      2|none  |     0|acc   |↑  |0.4338|±  |0.0142|

# 8B
|    Tasks     |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------|------:|------|-----:|------|---|-----:|---|-----:|
|truthfulqa_mc2|      2|none  |     0|acc   |↑  |0.5402|±  | 0.015|

# 1B ref
|    Tasks     |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------|------:|------|-----:|------|---|-----:|---|-----:|
|truthfulqa_mc2|      2|none  |     0|acc   |↑  |0.4392|±  |0.0144|

