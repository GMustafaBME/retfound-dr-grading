# RETFound Fine-Tuning for Diabetic Retinopathy Grading

Fine-tuning [RETFound](https://github.com/rmaphoh/RETFound_MAE) (ViT-Large, pre-trained on 1.6M retinal images) for automated diabetic retinopathy severity grading from fundus photographs.

## Key Results

| Model | AUROC | Sensitivity | Specificity | Kappa |
|-------|-------|-------------|-------------|-------|
| RETFound (ours) | 0.XX | 0.XX | 0.XX | 0.XX |
| EfficientNet-B7 | 0.XX | 0.XX | 0.XX | 0.XX |
| ResNet-50 | 0.XX | 0.XX | 0.XX | 0.XX |

> Replace with actual results when available

## Architecture

Input Fundus Image (224x224)
|
RETFound ViT-Large Encoder
(pre-trained on 1.6M retinal images)
|
[CLS] Token Embedding
|
Classification Head (5-class DR grading)
|
Output: {No DR, Mild, Moderate, Severe, Proliferative}

## Features
- Layer-wise learning rate decay for stable foundation model fine-tuning
- Progressive unfreezing strategy
- Grad-CAM explainability for clinical interpretability
- Set-valued prediction for uncertainty quantification
- Benchmarked on Messidor-2 and local clinical dataset (550+ images)

## Quick Start

```bash
git clone https://github.com/GMustafaBME/retfound-dr-grading.git
cd retfound-dr-grading
pip install -r requirements.txt
python src/train.py --config configs/default.yaml
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```

## Project Structure
retfound-dr-grading/
├── src/
│   ├── data_loader.py    # Dataset loading & augmentation
│   ├── model.py          # RETFound architecture & config
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Metrics & evaluation
│   └── gradcam.py        # Explainability visualizations
├── configs/
│   └── default.yaml      # Hyperparameter configuration
├── requirements.txt
├── Dockerfile
└── README.md

## Related Publication

Wu, H., [...], **Mustafa, G.**, & Fang, Y. (2025). Multi-Scale Target-Aware Representation Learning for Fundus Image Enhancement. *Neural Networks (Elsevier)*, Vol. 195.

## Author
Ghulam Mustafa | Peking University | mustafabme@gmail.com

## License
MIT
