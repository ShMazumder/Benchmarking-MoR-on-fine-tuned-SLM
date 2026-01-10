# Quick Start Guide - Bangla Training Final Fix

## 📋 Files Created

1. **`run_bangla_FINAL_FIX.ipynb`** ⭐ - Main notebook (USE THIS!)
2. **`train_amp_v2.py`** - Aggressive training script
3. **`download_curated_bangla.py`** - Corpus downloader
4. **`AGGRESSIVE_FIXES_GUIDE.md`** - Detailed documentation

---

## 🚀 Quick Start (3 Steps)

### Step 1: Upload to Kaggle/Colab
Upload `run_bangla_FINAL_FIX.ipynb` to:
- **Kaggle**: Enable GPU P100
- **Colab**: Runtime → GPU

### Step 2: Run All Cells
Just click "Run All" - it handles everything:
- ✅ Downloads curated corpus
- ✅ Applies aggressive config
- ✅ Trains all models
- ✅ Plots results

### Step 3: Check Results
Look for:
- ✅ **N=6**: >30% accuracy
- ✅ **N=12**: >15% accuracy
- ✅ **Loss**: Decreasing steadily

---

## 🎯 What's Different?

### Before (Failed):
```
Vocab: 4K
Epochs: 2
LR: 1e-3 (fixed)
Batch: 128
Result: 3% accuracy (stuck)
```

### After (Fixed):
```
Vocab: 16K ✅
Epochs: 10 ✅
LR: 1e-4 (warmup + cosine) ✅
Batch: 64 ✅
Gradient Clipping: 1.0 ✅
Weight Decay: 0.01 ✅
Corpus: Curated (IndicNLP) ✅
Result: 15-30% accuracy (expected)
```

---

## 📊 Expected Timeline

| Task | Time | GPU |
|------|------|-----|
| Download corpus | 5 min | - |
| Train N=6 | 45 min | P100 |
| Train N=12 | 90 min | P100 |
| Train MoR (2x) | 180 min | P100 |
| **Total** | **~5 hours** | P100 |

---

## ✅ Success Indicators

**During Training:**
```
Epoch 1: Loss=8.14, Acc=3.67%
Epoch 2: Loss=7.85, Acc=5.12%  ← Good! Decreasing
Epoch 3: Loss=7.52, Acc=7.89%  ← Good! Increasing
Epoch 5: Loss=6.91, Acc=12.45% ← Excellent!
Epoch 10: Loss=6.23, Acc=18.67% ← Success!
```

**Bad Signs:**
```
Epoch 1: Loss=8.14, Acc=3.67%
Epoch 2: Loss=8.13, Acc=3.68%  ← Bad! Flat
Epoch 3: Loss=8.12, Acc=3.67%  ← Bad! No learning
```

If you see flat loss after 3 epochs, **STOP** and:
1. Reduce LR to `5e-5` in `train_amp_v2.py`
2. Reduce batch to `32` in `config.py`

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: datasets"
```bash
!pip install datasets sentencepiece
```

### Issue: "CUDA out of memory"
Reduce batch size in `config.py`:
```python
batch_size = 32  # Was: 64
```

### Issue: Still stuck at 3% after 3 epochs
Try even lower LR in `train_amp_v2.py`:
```python
base_lr = 5e-5  # Was: 1e-4
```

### Issue: N=12 fails but N=6 works
**This is scientifically valid!**
- Shows deep models need more data
- Keep honest limitations in manuscript
- Focus on N=6 results

---

## 📝 What to Report in Manuscript

### If Fixes Work (>15% accuracy):
```
"After implementing aggressive optimizations (LR warmup, 
cosine annealing, gradient clipping, curated corpus), 
the Bangla experiments achieved X% accuracy, validating 
the approach across morphologically rich languages."
```

### If Still Fails (<15% accuracy):
```
"Despite aggressive optimizations (16K vocab, 10 epochs, 
LR scheduling, curated corpus), deep models (N=12) 
struggled on the small Bangla corpus (X% accuracy), 
while shallow models (N=6) achieved Y% accuracy. 
This demonstrates the fundamental challenge of training 
deep architectures on limited low-resource language data."
```

**Both are publishable!** Science values truth, not just success.

---

## 📂 File Structure

```
code/
├── run_bangla_FINAL_FIX.ipynb    ← RUN THIS
├── train_amp_v2.py               ← Aggressive training
├── download_curated_bangla.py    ← Corpus downloader
├── AGGRESSIVE_FIXES_GUIDE.md     ← Full docs
└── data/
    └── bangla/
        └── bangla_curated.txt    ← Auto-downloaded
```

---

## 🎓 Key Learnings

1. **Vocab size matters** - 4K too small for Bangla
2. **LR scheduling critical** - Fixed LR causes stuck training
3. **Gradient clipping essential** - Deep models need it
4. **Data quality > quantity** - Curated corpus helps
5. **Deep ≠ always better** - N=6 may outperform N=12 on small data

---

## 🚨 Emergency Contact

If nothing works after all fixes:
1. Share training logs (first 5 epochs)
2. Check if loss is NaN (numerical instability)
3. Try N=4 or N=3 (even shallower)
4. Consider this a "negative result" (still valuable!)

---

**Good luck! 🍀**
