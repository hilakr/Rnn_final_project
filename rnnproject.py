import itertools
import nltk
import sys
import os
import theano
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
import re, math
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "start"
sentence_end_token = "end"

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print ("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            # if (len(losses) &gt; 1 and losses[-1][1] &gt; losses[-2][1]):
            #     learning_rate = learning_rate * 0.5
            #     print ("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


#open the file and split it to sentences
with open("phpcode3.txt", 'rb') as f:
    sentences = [line.strip() for line in f]
    print (sentences)
    # Append SENTENCE_START and SENTENCE_END
    #sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent.decode()) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found %d unique words tokens." % len(word_freq.items()))


# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
print(word_to_index)
print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

# Print an training data example
x_example, y_example = X_train[17], y_train[17]
print ("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
print ("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))

#Training our Network with Theano and the GPU

# To avoid performing millions of expensive calculations we use a smaller vocabulary size for checking.
grad_check_vocab_size = 5
model = RNNTheano(grad_check_vocab_size, 10)
gradient_check_theano(model, [0,1,2,3], [1,2,3,4])

np.random.seed(10)
model = RNNTheano(vocabulary_size)
model.sgd_step(X_train[10], y_train[10], 0.005)


# Run the model
model = RNNTheano(vocabulary_size, hidden_dim=50)
load_model_parameters_theano('./data/trained-model-theano.npz', model)

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


num_sentences = 10
senten_min_length = 7
string = 'new_sentence: '
new_sentences = []
for i in range(num_sentences):
    print(string + str(i))
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    new_sentences.append(sent)
    print (" ".join(sent))

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

plot_point = []
print (range(len(new_sentences)))
temp = 0;
for i in range(len(new_sentences)):
    text1 = ''.join(str(e) for e in new_sentences[i])
    for j in range(len(tokenized_sentences)):
        text2 = ''.join(str(e) for e in tokenized_sentences[j])
        vector1 = text_to_vector(text1)
        vector2 = text_to_vector(text2)
        if temp < get_cosine(vector1, vector2):
            temp = get_cosine(vector1, vector2)
    plot_point.append(temp)

print("similarity average by cosine similarity:" + np.mean(plot_point))
print(plot_point)
plt.plot(plot_point)
plt.ylabel('Cosine similarity')
plt.show()

plot_point2 = []
temp = 0
print (range(len(new_sentences)))
temp = 0;
for i in range(len(new_sentences)):
    text1 = ''.join(str(e) for e in new_sentences[i])
    set_sentence1 = set(text1.split())
    for j in range(len(tokenized_sentences)):
        text2 = ''.join(str(e) for e in tokenized_sentences[j])
        set_sentence2 = set(text2.split())
        similarity = (1.0 + len(set_sentence1.intersection(set_sentence2))) / (1.0 + max(len(set_sentence1), len(set_sentence2)))
        if temp < similarity:
            temp = similarity
    plot_point2.append(temp)

print(plot_point2)
print("similarity average by SET similarity:" + np.mean(plot_point2))
plt.plot(plot_point2)
plt.ylabel('set similarity')
plt.show()

# from sklearn.feature_extraction.text import TfidfVectorizer
#
# plot_point3 = []
# for x in range(len(new_sentences)):
#     myDocs =  []
#     myDocs.append(new_sentences[x]);
#     myDocs.extend(tokenized_sentences)
#     tfidf_vectorizer = TfidfVectorizer()
#     tfidf_matrix = tfidf_vectorizer.fit_transform(myDocs)
#     plot_point3.append(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))
#
# print(plot_point3)
# plt.plot(plot_point3)
# plt.ylabel('cosine similarity using tf-idf')
# plt.show()