# Generating Qualitative Visualizations with Real Data

This guide explains how to generate the qualitative examples figure using real ICH artifact images.

## Step 1: Prepare Real Dataset

### Option A: Download from Public Sources

1. **Create dataset structure:**
```bash
python scripts/download_ich_data.py --data_dir ./data/ich_benchmark --mode structure
```

2. **Download real images** from sources like:
   - Wikimedia Commons (CC-licensed ICH images)
   - Museum digital collections (with proper attribution)
   - UNESCO ICH lists
   - Open cultural heritage datasets

3. **Organize images** in the directory structure:
```
data/ich_benchmark/
├── train/
│   ├── textile/
│   │   ├── northern/
│   │   │   └── *.jpg
│   │   ├── southern/
│   │   └── ...
│   ├── ceramic/
│   └── ...
├── val/
└── test/
```

### Option B: Use Existing Dataset

If you have an existing ICH dataset, organize it according to the structure above.

## Step 2: Train the Model

Train DAPN on your real dataset:

```bash
python scripts/train.py \
    --data_dir ./data/ich_benchmark \
    --save_dir ./checkpoints \
    --epochs 100 \
    --n_way 5 \
    --k_shot 1
```

This will save the best model checkpoint.

## Step 3: Generate Qualitative Visualization

Generate the qualitative examples figure using the trained model and real images:

```bash
python scripts/generate_qualitative_visualization.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data/ich_benchmark \
    --split test \
    --output ../qualitative_examples_real.pdf \
    --num_samples 5
```

## What the Script Does

1. **Loads trained model** from checkpoint
2. **Loads real images** from test dataset
3. **Samples 5 examples** (one per category)
4. **Extracts features** using the trained model:
   - Domain-invariant features (row 2)
   - Domain-specific features (row 3)
5. **Computes attention maps** using Grad-CAM to visualize what the model focuses on
6. **Generates visualization** showing:
   - Original images with predictions (row 1)
   - Domain-invariant feature visualizations (row 2)
   - Domain-specific feature visualizations (row 3)

## Output

The script generates a PDF figure (`qualitative_examples_real.pdf`) showing:
- **Row 1**: Sample ICH artifact images with predictions
- **Row 2**: Domain-invariant features (what model focuses on for classification)
- **Row 3**: Domain-specific features (variations that are ignored)

## Requirements

- Trained DAPN model checkpoint
- Real ICH dataset with organized structure
- scipy (for image resizing in Grad-CAM): `pip install scipy`

## Notes

- The visualization uses Grad-CAM to highlight important regions
- Feature intensity percentages are computed from feature magnitudes
- Colors: Blue for domain-invariant, Orange for domain-specific
- The figure follows Nature journal style for publication quality

## Troubleshooting

If images don't load:
- Check dataset directory structure
- Ensure images are in JPEG/PNG format
- Verify image paths are correct

If model predictions are poor:
- Ensure model is fully trained (check validation accuracy)
- May need to fine-tune on your specific dataset
- Check that dataset categories match training categories
