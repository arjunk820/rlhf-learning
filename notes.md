Following this HF tutorial: https://huggingface.co/docs/trl/sft_trainer

## SFT Method

SFT trains language models to target datasets. The model is trained in a
fully supervised fashion using pairs of inputs and outputs.

We want to minimize the negative log-likelihood (NLL).