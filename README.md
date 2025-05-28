# Naive Bayes Spam Filter
A Python Naive Bayes spam filter trained on raw email data with support for tokenization, smoothing, and indicative word analysis.

## 📂 Project Structure

- `load_tokens()` — Parses email files into tokens using Python’s built-in `email` module.
- `log_probs()` — Computes Laplace-smoothed log-probabilities for all vocabulary terms.
- `SpamFilter` class:
  - `__init__()` — Initializes the filter using spam and ham training directories.
  - `is_spam()` — Predicts whether an email is spam using log-space likelihoods.
  - `most_indicative_spam()` / `most_indicative_ham()` — Returns the most spammy/hammy words based on information gain.

## 🔍 How it Works

The classifier is trained on two directories:
- `spam/` — directory of spam emails
- `ham/` — directory of ham (non-spam) emails

It builds log-probability dictionaries for each class using token frequency and Laplace smoothing. At test time, it computes log-likelihoods for each email and predicts the more probable class.

The classifier also calculates the **most indicative words** for spam and ham by computing:

```
log(P(w | class) / P(w))
```

## 🧪 Example Usage

```python
sf = SpamFilter("data/train/spam", "data/train/ham", 1e-5)
sf.is_spam("data/dev/email123")  # returns True or False

sf.most_indicative_spam(5)  # returns top 5 spam words
sf.most_indicative_ham(5)   # returns top 5 ham words
```

## 📁 Dependencies

This project uses only Python’s standard library:
- `math`
- `os`
- `collections.Counter`
- `email`