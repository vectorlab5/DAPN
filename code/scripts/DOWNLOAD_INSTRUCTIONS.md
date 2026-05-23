# Instructions for Downloading Real ICH Datasets

## Overview

This guide explains how to download real Intangible Cultural Heritage (ICH) images from public museum APIs and organize them for DAPN experiments.

## Method 1: Metropolitan Museum of Art (Recommended - Easiest)

The Met Museum provides **free, public domain (CC0) images** via their API.

### Quick Start:

```bash
cd code/scripts
python download_real_ich_data.py --source metmuseum --max_per_category 100
```

This will:
1. Download images from 5 ICH categories
2. Organize them into train/val/test splits
3. Assign domain labels (distributed evenly)

### What Gets Downloaded:

- **Textile**: Tapestries, embroidery, fabrics
- **Ceramic**: Pottery, porcelain, vases
- **Wood**: Wood carvings, sculptures, furniture
- **Paper**: Paper art, calligraphy, scrolls
- **Metalwork**: Bronze, copper, silver artifacts

### Expected Results:

- ~100 images per category = ~500 total images
- Images organized as: `data/ich_benchmark/{train,val,test}/{category}/{domain}/*.jpg`
- Metadata JSON files saved alongside images

## Method 2: Rijksmuseum API

Rijksmuseum (Netherlands) also provides open data:

1. **Get free API key**: https://www.rijksmuseum.nl/en/api
2. **Download script**: (can be added to download_real_ich_data.py)
3. **Use API key**: `--api_key YOUR_KEY`

## Method 3: Few-Shot Learning Benchmarks

For comparison experiments, download standard benchmarks:

### mini-ImageNet:

```bash
# Option 1: Using torchmeta (recommended)
pip install torchmeta
python download_real_ich_data.py --source miniimagenet

# Option 2: Manual download
# 1. Download ImageNet (ILSVRC2012)
# 2. Download split CSVs from:
#    https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet
# 3. Organize according to splits
```

### CIFAR-FS:

```bash
# Download from: https://github.com/bertinetto/r2d2/tree/master/data
# Or use: pip install torchmeta
```

## Method 4: Manual Collection

For domain-specific ICH collections:

1. **Chinese Cultural Heritage**: 
   - Visit museum websites
   - Use digital collections (check licensing)
   - Organize manually

2. **UNESCO ICH Lists**:
   - Text-based lists with limited images
   - May need manual image collection

3. **Academic Collections**:
   - Contact researchers/institutions
   - Check for open access datasets

## Organization Structure

After download, images should be organized as:

```
data/ich_benchmark/
├── train/
│   ├── textile/
│   │   ├── northern/
│   │   │   └── *.jpg
│   │   ├── southern/
│   │   ├── eastern/
│   │   └── western/
│   ├── ceramic/
│   ├── wood/
│   ├── paper/
│   └── metalwork/
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

## Domain Assignment

Currently, domains are assigned automatically (round-robin distribution). For better results:

1. Use metadata JSON files to assign domains based on:
   - Geographic origin (culture, period fields)
   - Time period
   - Style characteristics

2. Manually review and reassign if needed

## Verification

Check downloaded dataset:

```bash
# Count images
find data/ich_benchmark -name "*.jpg" | wc -l

# Check structure
tree data/ich_benchmark -L 3

# Verify with Python
python -c "from code.data import ICHDataset; d = ICHDataset('data/ich_benchmark', 'test'); print(f'{len(d)} images, {len(set(d.labels))} classes')"
```

## Training After Download

Once real data is downloaded:

```bash
# Train on real data
cd code
python scripts/train.py \
    --data_dir ../data/ich_benchmark \
    --epochs 100 \
    --batch_size 16

# Generate visualization
python scripts/generate_qualitative_visualization.py \
    --checkpoint ../checkpoints/best_model.pth \
    --data_dir ../data/ich_benchmark \
    --split test \
    --output ../figures/qualitative_examples_real.pdf
```

## Notes

- **Licensing**: Met Museum images are CC0 (public domain), safe for publication
- **Quality**: Review downloaded images - some may not be relevant
- **Metadata**: JSON files contain title, culture, period - useful for domain assignment
- **Rate Limits**: API has rate limits - script includes delays
- **Size**: Full dataset may take time to download (100s of images)

## Troubleshooting

**No images downloaded:**
- Check internet connection
- Verify API is accessible
- Try with smaller `--max_per_category`

**Wrong categories:**
- Adjust search terms in script
- Manually filter after download

**Domain assignment issues:**
- Review metadata JSON files
- Manually reassign domains
- Use culture/period fields to guide assignment
