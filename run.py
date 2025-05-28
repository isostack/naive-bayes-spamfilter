import math 
from collections import Counter
import os
import email
from email.iterators import body_line_iterator
from email import policy
from email.parser import BytesParser  

def load_tokens(email_path): #1
    list = []

    # Read email
    with open(email_path, 'r', encoding='utf-8', errors='ignore') as file:
        msg = email.message_from_file(file, policy=policy.default)
        # extracts the tokens from its message
        for line in body_line_iterator(msg):
            token = line.split()
            list.extend(token)

    return list


def log_probs(email_paths, smoothing): #2
    # P(w) = (count(w) + α) / (∑_{w'∈V} count(w') + α(|V| + 1))
    # P(<UNK>) = α / (∑_{w'∈V} count(w') + α(|V| + 1))
                                   
    counts = Counter()

    for path in email_paths:
        words = load_tokens(path)  # get list of words from email
        counts.update(words)

    # Find totals
    total_words = sum(counts.values())
    vocab = list(counts.keys())
    vocab_size = len(vocab)

    # (total count of all words + smoothing * (vocab size + 1 for <UNK>))
    denominator = total_words + smoothing * (vocab_size + 1)

    log_probs = {}
    for word in vocab:
        count = counts[word]
        prob = (count + smoothing) / denominator
        log_probs[word] = math.log(prob)
    unk_prob = smoothing / denominator
    log_probs["<UNK>"] = math.log(unk_prob)

    return log_probs

class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing): #3
        self.smoothing = smoothing
        
        # Create spam and ham arrays
        self.spam_files = []
        for filename in os.listdir(spam_dir):
            self.spam_files.append(os.path.join(spam_dir, filename))

        self.ham_files = []
        for filename in os.listdir(ham_dir):
            self.ham_files.append(os.path.join(ham_dir, filename))

        # COUNT # emails
        num_spam = len(self.spam_files)
        num_ham = len(self.ham_files)
        total = num_spam + num_ham

        # Create 2 log-probability dictionaries corresponding to the emails
        # -SPAM
        self.log_spam_prior = math.log(num_spam / total)
        self.spam_log_probs = log_probs(self.spam_files, smoothing)
        self.spam_vocab = set(self.spam_log_probs.keys())

        # -HAM
        self.log_ham_prior = math.log(num_ham / total)
        self.ham_log_probs = log_probs(self.ham_files, smoothing)
        self.ham_vocab = set(self.ham_log_probs.keys())

        # COMBINE
        self.all_vocab = self.spam_vocab.union(self.ham_vocab)

    def is_spam(self, email_path): #4
        words = load_tokens(email_path)
        counts = Counter(words)

        # Get spam totals
        spam_total = self.log_spam_prior
        ham_total = self.log_ham_prior

        for word, count in counts.items():
            # Spams
            spam_log_prob = self.spam_log_probs.get(word, self.spam_log_probs["<UNK>"])
            spam_total += count * spam_log_prob

            # Hams
            ham_log_prob = self.ham_log_probs.get(word, self.ham_log_probs["<UNK>"])
            ham_total += count * ham_log_prob

        # OUTPUT SPAM?
        if spam_total > ham_total:
            # YES SPAM
            return True
        else:
            # NOT SPAM (HAM)
            return False

    def most_indicative_spam(self, n): #5
        scores = []

        # GO THROUGH WORDS
        for word in self.all_vocab:
            # CHECK: word in spam and ham
            if (word not in self.spam_log_probs) or (word not in self.ham_log_probs):
                continue

            # Get log given spam and regular prob
            log_prob_X_spam = self.spam_log_probs[word]
            prob_X_spam = math.exp(log_prob_X_spam)

            # Get log given ham and regular prob
            log_prob_X_ham = self.ham_log_probs[word]
            prob_X_ham = math.exp(log_prob_X_ham)  

            # P(word) = P(word|spam)*P(spam) + P(word|ham)*P(ham)
            prob_word = (prob_X_spam * math.exp(self.log_spam_prior) + 
                         prob_X_ham * math.exp(self.log_ham_prior) )

            # log(P(w | spam) / P(w))
            score = log_prob_X_spam - math.log(prob_word)

            # save
            scores.append((score, word))

        scores.sort(reverse=True)
        outputs = []
        for pair in scores[:n]:
            _, word = pair
            outputs.append(word)

        return outputs

    def most_indicative_ham(self, n): #5
        scores = []

        # GO THROUGH WORDS
        for word in self.all_vocab:
            # CHECK: word in spam and ham
            if (word not in self.spam_log_probs) or (word not in self.ham_log_probs):
                continue

            # Get log given ham and regular prob
            log_prob_X_ham = self.ham_log_probs[word]
            prob_X_ham = math.exp(log_prob_X_ham)

            # Get log given spam and regular prob
            log_prob_X_spam = self.spam_log_probs[word]
            prob_X_spam = math.exp(log_prob_X_spam)

            # P(word) = P(word|ham)*P(ham) + P(word|spam)*P(spam)
            prob_word = (prob_X_ham * math.exp(self.log_ham_prior) +
                        prob_X_spam * math.exp(self.log_spam_prior) )

            # log(P(w | ¬spam) / P(w))
            # = log(P(w | ham) / P(w))
            score = log_prob_X_ham - math.log(prob_word)

            # save
            scores.append((score, word))

        # sort and return top n words
        scores.sort(reverse=True)
        outputs = []
        for pair in scores[:n]:
            _, word = pair
            outputs.append(word)

        return outputs

