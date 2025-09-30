```markdown
# Spam Detector (PyTorch + LSTM)

This project implements a text classification model to detect spam messages using PyTorch.  
The model is trained on an SMS spam dataset and allows interactive testing of new messages.

---

## Features

- Text preprocessing (cleaning, lowercasing, spam-indicator tokens)
- Vocabulary builder with frequency cutoff
- Pretrained GloVe embeddings
- Bidirectional LSTM classifier
- Training with early stopping and gradient clipping
- Evaluation with accuracy, precision, recall, and F1-score
- Interactive prediction loop for user input

---

## Project Structure
```

├── spam.csv # Dataset
├── main.py # Training, evaluation, and prediction
├── requirements.txt # Dependencies
└── README.md # Project overview

````

---

## Installation
```bash
git clone https://github.com/arunarun58/pytorch-spam-detector.git
cd spam-detector
pip install -r requirements.txt
````

---

Example:

```
Enter a message (or type 'exit' to quit): URGENT! You have won $1000. Claim now at http://scam.link
Prediction: spam (97.4%)
```

---

## Results

- Test Accuracy: approximately 95%
- Effective at distinguishing between spam and legitimate SMS messages
- Limited performance on phishing-style or sophisticated spam content

---

## Future Work

- Expand the dataset to include phishing emails
- Experiment with transformer-based models (e.g., DistilBERT, BERT)
- Incorporate subword tokenization for improved handling of rare words

---

## License

MIT
