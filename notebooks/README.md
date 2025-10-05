# Notebooks

This folder contains Jupyter notebooks for running the RLHF learning experiments.

## Available Notebooks

### `sft_training_colab.ipynb`
A Google Colab-compatible notebook for Supervised Fine-Tuning (SFT) of GPT-2 for tweet generation.

**Features:**
- 🚀 **GPU Support**: Automatically detects and uses available GPUs (CUDA/MPS/CPU)
- 📊 **W&B Integration**: Real-time experiment tracking with Weights & Biases
- 🎯 **Optimized for Colab**: Memory-efficient settings and cloud storage support
- 🧪 **Model Testing**: Built-in functions to test your trained model

## How to Use

### Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `sft_training_colab.ipynb` to Colab
3. Enable GPU runtime: Runtime → Change runtime type → GPU
4. Run the cells sequentially
5. Upload your `tweet_sft_dataset_10k.jsonl` file when prompted (or use sample data)

## Prerequisites

- Python 3.8+
- Required packages (installed automatically in Colab)
- W&B account for experiment tracking (optional but recommended)

## Data Requirements

The notebook expects a JSONL file with the following format:
```json
{"instruction": "Write a personal_story tweet about coding", "response": "Spent 2 hours debugging a typo. It was a missing semicolon 😅"}
{"instruction": "Write a classic tweet about wisdom", "response": "The most dangerous phrase in programming: 'Just a small change'"}
```

## Output

The training will generate:
- Model checkpoints in `./sft_results/`
- Training logs in `./logs/`
- W&B experiment tracking (if configured)
- Test results from the trained model

## Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch size or use smaller model
2. **W&B Login Issues**: Make sure you have a valid API key
3. **Dataset Not Found**: Upload your dataset file or use sample data
4. **GPU Not Available**: Check Colab runtime settings

### Performance Tips:
- Use GPU runtime in Colab for faster training
- Enable mixed precision (fp16) for memory efficiency
- Monitor W&B dashboard for real-time metrics
- Save checkpoints regularly to avoid losing progress
