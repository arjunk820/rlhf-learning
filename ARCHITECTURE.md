# Architecture: Learning SFT for Viral Tweets

## Goals & Scope

* **Primary Goal:** Learn Supervised Fine-Tuning (SFT) concepts hands-on
* **Secondary Goal:** Generate viral-style tweets and track what works
* **Approach:** Simple, iterative experimentation without over-engineering
* **Model:** GPT-2 (base/medium) with basic SFT
* **Data:** 10k instruction-response pairs with engagement metadata

---

## Data Requirements

### Target Dataset Size
- **10,000 instruction-response pairs** for effective SFT training
- Mix of high and medium engagement tweets
- Balanced across topics and styles

### Required Metadata
Each tweet must include:
- **likes** (number)
- **comments** (number) 
- **impressions** (number)

### Data Sources
1. **Your tweets** (primary style baseline)
2. **Viral tech accounts** (engagement patterns)
3. **Diverse tech topics** (generalizability)

### Data Format
```json
{
  "instruction": "Write a witty tweet about debugging",
  "response": "Just spent 3 hours debugging a typo. The variable was called 'userName' but I was using 'username' 😅",
  "likes": 1567,
  "comments": 89,
  "impressions": 12000,
  "topic": "debugging",
  "style": "humor"
}
```

---

## Simplified Training Pipeline

### 1. Data Collection
- **Today:** Create sample dataset (10k examples) to understand SFT
- **Tomorrow:** Collect real tweet data using scraping tools
- **Format:** Convert to instruction-response pairs with engagement weighting

### 2. SFT Training
- **Model:** GPT-2 base or medium
- **Framework:** Hugging Face Transformers + TRL
- **Loss:** Standard cross-entropy with engagement weighting
- **Evaluation:** Generate sample tweets and assess quality

### 3. Experiment Tracking
- **Tool:** Weights & Biases (simple logging)
- **Metrics:** Training loss, generated tweet quality
- **Iterations:** Try different hyperparameters and data mixes

---

## Implementation Plan

### Phase 1: Learning SFT (Today)
- Create sample tweet dataset
- Set up basic SFT training
- Understand the concepts hands-on

### Phase 2: Real Data (Tomorrow)
- Collect actual tweet data
- Scale up to 10k examples
- Improve training pipeline

### Phase 3: Experimentation (This Week)
- Try different model sizes
- Experiment with engagement weighting
- Test various prompt formats

### Phase 4: Evaluation (Next Week)
- Generate and evaluate tweets
- Document what works
- Plan next steps

---

## Tech Stack (Minimal)

* **Data:** Python, pandas, json
* **Modeling:** transformers, trl, torch
* **Tracking:** wandb
* **Scraping:** snscrape or alternatives

---

## Success Metrics

1. **Learning:** Understand SFT concepts and implementation
2. **Functionality:** Generate coherent, tweet-like text
3. **Quality:** Generated tweets feel natural and engaging
4. **Experimentation:** Track what works and what doesn't

---

**Focused on:**
- Core SFT learning
- Simple data collection
- Basic experiment tracking
- Iterative improvement
