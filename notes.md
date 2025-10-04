# Supervised Fine-Tuning (SFT) Learning Notes

Following this HF tutorial: https://huggingface.co/docs/trl/sft_trainer

## SFT Fundamentals

### What is SFT?
SFT trains language models to target datasets. The model is trained in a
fully supervised fashion using pairs of inputs and outputs.

As it states in the name, our dataset is labelled with the "correct" target.
Ultimately, we want our model to predict the next token as well as possible.


### Objective Function
We want to minimize the negative log-likelihood (NLL):
```
Loss = -∑ log P(y_i | x_i, θ)
```
Where:
- `x_i` = input sequence
- `y_i` = target output sequence  
- `θ` = model parameters

Based on the input `x_i`, we want the model to predict `y_i` --> our dataset
has "correct" `y_i`. We use log likelihood to transform the probabilities
into more manageable numbers and for mathematical convenience. Their meaning remains the same.

Low NLL means higher probability. Since the log operation negates, we negate the log
operation to keep positivity.

.22 NLL ~ .8 probability
2.3 NLL ~ .1 probability

The NLL values are more distinct, which is easier for understanding and 
optimization.

### Key Concepts

1. **Teacher Forcing**: During training, we provide the model with correct previous tokens when predicting the next token, not its own predictions.

2. **Data Format**: SFT requires input-output pairs in the form of instruction
and response:
   ```
   Input: "Instruction: What is the capital of France?"
   Output: "Response: The capital of France is Paris."
   ```

3. **Training Process**:
   - Load pre-trained model (e.g., GPT-2)
   - Prepare instruction-response dataset
   - Fine-tune using supervised learning
   - Model learns to generate appropriate responses

### Implementation Notes

- Using TRL (Transformer Reinforcement Learning) library for SFT
- GPT-2 as base model (good balance of size and capability)
- Sample dataset with instruction-following examples
- Weights & Biases for experiment tracking

We can't train on my computer regardless of using MPS.

Some options:
- Cloud Training
- External Storage
- Smaller Dataset (currently at 10k examples)
- Model Quantization