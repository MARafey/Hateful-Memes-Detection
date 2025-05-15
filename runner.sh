# Train all models (default 10 epochs)
python src/run_all.py --train

# Evaluate trained models
python src/run_all.py --evaluate

# Launch the web interface for inference
python src/run_all.py --interface

# Train only specific models by skipping others
python src/run_all.py --train --skip_models text_only image_only

# Run with custom epochs and learning rate
python src/run_all.py --train --epochs 20 --learning_rate 1e-5

# Generate confusion matrices and model architecture reports
python src/model_evaluation.py --output_dir "Confusion Matrix"