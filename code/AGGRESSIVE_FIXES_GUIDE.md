# Aggressive Training Fixes - Implementation Guide

## Problem Summary
The Bangla experiments failed even with 16K vocabulary and 10 epochs:
- **Loss stuck at 8.14** (no learning)
- **Accuracy flat at 3.67%** (random guessing for 16K classes)
- Root cause: **Optimization failure**, not just data quality

## Solutions Implemented

### 1. **Curated Bangla Corpus** (`download_curated_bangla.py`)

**Why the change?**
- Raw Wikipedia has noise, formatting issues, mixed languages
- Need high-quality, cleaned, deduplicated text

**What we use:**
- **Primary**: `ai4bharat/IndicNLPSuite` (bn) - Professional-grade corpus
- **Fallback**: Cleaned Wikipedia with aggressive filtering

**Features:**
- ✅ Removes URLs, emails, non-Bangla text
- ✅ Filters lines with <60% Bangla characters
- ✅ Deduplicates sentences
- ✅ Proper sentence segmentation
- ✅ 20MB curated text (vs 15MB raw)

**Usage:**
```bash
python download_curated_bangla.py
# Creates: data/bangla/bangla_curated.txt
```

---

### 2. **Aggressive Training Script** (`train_amp_v2.py`)

**Key Changes:**

#### A. **Lower Base Learning Rate**
```python
base_lr = 1e-4  # Was: 3e-4
```
- **Why**: High LR causes model to overshoot optimal weights
- **Impact**: Smoother convergence, better final loss

#### B. **Learning Rate Warmup + Cosine Annealing**
```python
warmup_steps = int(0.1 * total_steps)  # 10% warmup
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
```
- **Warmup**: Gradually increase LR from 0 → base_lr
- **Cosine**: Smoothly decrease LR to near-zero
- **Why**: Prevents early instability, improves final convergence

#### C. **Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- **Why**: Prevents exploding gradients in deep models
- **Impact**: Training stability

#### D. **Weight Decay Regularization**
```python
optimizer = optim.AdamW(lr=base_lr, weight_decay=0.01)
```
- **Why**: Prevents overfitting, improves generalization
- **Impact**: Better test accuracy

#### E. **Better Optimizer Settings**
```python
betas=(0.9, 0.999), eps=1e-8
```
- **Why**: Standard best practices for Adam

---

## Expected Improvements

### Before (Original):
| Metric | Value | Status |
|--------|-------|--------|
| Vocab Size | 4K | Too small |
| Epochs | 2 | Insufficient |
| Learning Rate | 1e-3 (fixed) | Too high |
| Loss | 7.2 → 7.2 | Stuck |
| Accuracy | 3% → 3% | No learning |

### After (Improved):
| Metric | Value | Expected |
|--------|-------|----------|
| Vocab Size | 16K | ✅ Better |
| Epochs | 10 | ✅ Sufficient |
| Learning Rate | 1e-4 (scheduled) | ✅ Optimal |
| Loss | 8.1 → **<6.0** | ✅ Learning |
| Accuracy | 3.67% → **>15%** | ✅ Progress |

**Realistic Target for Bangla N=12:**
- **Loss**: Should drop below 6.0 by epoch 10
- **Accuracy**: 15-25% (vs 3% before)
- **N=6 Baseline**: Should reach 30-35% (vs 25% before)

---

## How to Use

### Step 1: Download Curated Corpus
```bash
cd code
python download_curated_bangla.py
```

### Step 2: Update Data Loader
Edit `data/bangla.py`:
```python
# Change this line:
BANGLA_DATA_PATH = Path('data/bangla/bangla_slm.txt')
# To:
BANGLA_DATA_PATH = Path('data/bangla/bangla_curated.txt')
```

### Step 3: Run with Aggressive Training
```bash
python train_amp_v2.py \
  --dataset bangla \
  --experiment baseline_12 \
  --tokenization subword \
  --subword_vocab_size 16000 \
  --epochs 10 \
  --device cuda \
  --amp
```

---

## Monitoring Training

### Good Signs ✅:
1. **Loss decreases steadily** (8.1 → 7.5 → 7.0 → 6.5...)
2. **Accuracy increases** (3% → 5% → 8% → 12%...)
3. **Learning rate decreases** (1e-4 → 5e-5 → 1e-5...)
4. **No NaN/Inf values**

### Bad Signs ❌:
1. **Loss stays flat** (8.1 → 8.1 → 8.1...)
2. **Accuracy stuck** (3% → 3% → 3%...)
3. **Loss explodes** (8.1 → 15.0 → NaN)

If you see bad signs after 3 epochs, **stop and reduce LR further** to 5e-5.

---

## Fallback Plan

If aggressive fixes still don't work:

### Option A: Use N=6 Instead of N=12
```bash
python train_amp_v2.py --experiment baseline_6 ...
```
- Shallow models are easier to optimize
- Should reach 30-35% accuracy

### Option B: Use Smaller Batch Size
Edit `config.py`:
```python
batch_size = 64  # Was: 128
```
- Smaller batches = noisier gradients = better exploration

### Option C: Even Lower LR
Modify `train_amp_v2.py`:
```python
base_lr = 5e-5  # Was: 1e-4
```

---

## Files Created

1. **`train_amp_v2.py`** - Aggressive training script
2. **`download_curated_bangla.py`** - Curated corpus downloader
3. **`AGGRESSIVE_FIXES_GUIDE.md`** - This file

## Next Steps

1. ✅ Download curated corpus
2. ✅ Update data loader path
3. ⏳ Run training with `train_amp_v2.py`
4. ⏳ Monitor loss/accuracy progression
5. ⏳ Update manuscript with improved results (if successful)

---

## Scientific Integrity Note

If these fixes **still don't achieve >15% accuracy**:
- This is **valid scientific evidence**
- Shows deep models fundamentally struggle on small Bangla corpus
- Keep the honest limitations section in manuscript
- Consider this a "negative result" (still publishable!)

The goal is **truth**, not just good numbers. 🎯
