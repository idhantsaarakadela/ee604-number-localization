# Locating numbers in text images

EE604 project. A trained YOLOv8n model detects individual digits in a house-number crop. The inference pipeline reads the digits left to right and draws an enclosing box around the sequence.

## Run the demo

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use Python 3.11 on Linux with the packages listed in `packages.txt`. Upload a JPG or PNG or choose an included SVHN test crop. The model assumes one number per image. A stray detected digit can become part of the returned string. Confidence shown is the mean of the predicted digit confidences.

## Deploy

Following link can be used to run it on streamlit.
https://ee604-number-localization.streamlit.app/

## Repository contents

- `app.py` provides the browser interface.
- `inference_pipeline.py` and `weights/best.pt` are from the team model handover.
- `examples/` contains three SVHN test crops.
- `notebooks/` contains the dataset audit, literature review, evaluation audit, and report figure notebooks.
- The [project Drive](https://drive.google.com/drive/folders/1vDuDhDrIFKoXkZRtZLPvxyS63PofAZNF) holds the dataset, training handover, and saved experiment artifacts.

The cached validation baseline achieved 94.16% sequence localization F1, 84.08% exact text accuracy, and 83.72% joint accuracy at IoU ≥ 0.5. These are cached validation results. The browser demo uses the handover checkpoint with confidence 0.25, NMS IoU 0.65, and image size 640.

Data: [Street View House Numbers](https://ufldl.stanford.edu/housenumbers/).
