from typing import List, Tuple
from sklearn.model_selection import train_test_split
import math
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_train_data(
    positive_filepath: str,
    negative_filepath: str
) -> Tuple[List[str], List[int]]:
    """Load the training data, producing Lists of text and labels

    Args:
        filepath (str): Path to the training file

    Returns:
        Tuple[List[str], List[int]]: The texts and labels
    """

    def _read(filename: str):
        texts = []
        with open(filename,"r") as f:
            for line in f:
                _id, text = line.rstrip().split("\t")
                texts.append(text)

        return texts

    texts = []
    labels = []
    for text in _read(positive_filepath):
        texts.append(text)
        labels.append(1)

    for text in _read(negative_filepath):
        texts.append(text)
        labels.append(0)

    return texts, labels


def load_test_data(filepath: str) -> List[str]:
    """Load the test data, producing a List of texts

    Args:
        filepath (str): Path to the training file

    Returns:
        List[str]: The texts
    """
    texts = []
    labels = []
    with open(filepath, "r") as file:
        for line in file:
            idx, text, label = line.rstrip().split("\t")
            texts.append(text)
            if label == 'POS':
                label = 1
            else:
                label = 0
            labels.append(label)

    return texts, labels


def split_data( all_texts, all_labels):
    train_texts, dev_texts, train_labels, dev_labels = train_test_split(all_texts, all_labels, test_size= 0.2, random_state=42 )

    return train_texts, train_labels, dev_texts, dev_labels


def featurize_text(text):

    with open('positive-words.txt', "r", encoding="utf-8") as positive_words_file:
        positive_list = set(positive_words_file.read().splitlines())

    with open("negative-words.txt", "r", encoding="utf-8") as negative_words_file:
        negative_list = set(negative_words_file.read().splitlines())  

    pronouns = [
    "I", "me", "my", "mine", "myself",      
    "we", "us", "our", "ours", "ourselves", 
    "you", "your", "yours", "yourself", "yourselves" 
    ]
  
    x1=x2=x3=x4=x5=x6= 0
    words = text.lower().split()
    for word in words :
        if word in positive_list:
            x1 +=1
        elif word in negative_list:
            x2+= 1
        elif word == 'no':
            x3+=1
        elif word in pronouns:
            x4 +=1
        elif word == '!':
            x5+=1
        x6+=1

    x6 = math.log(x6)

    return [x1, x2, x3, x4, x5, x6]



def featurize_5_text(text):

    with open('positive-words.txt', "r", encoding="utf-8") as positive_words_file:
        positive_list = set(positive_words_file.read().splitlines())

    with open("negative-words.txt", "r", encoding="utf-8") as negative_words_file:
        negative_list = set(negative_words_file.read().splitlines())  

    
  
    x1=x2=x3=x5=x6= 0
    words = text.lower().split()
    for word in words :
        if word in positive_list:
            x1 +=1
        elif word in negative_list:
            x2+= 1
        elif word == 'no':
            x3+=1
        
        elif word == '!':
            x5+=1
        x6+=1

    x6 = math.log(x6)

    return [x1, x2, x3, x5, x6]



def normalize(train_vector):
    res = []
    for sublist in train_vector:
        maximum = max(sublist)
        minimum = min(sublist)
        normalized_sublist = []
        
        for x in sublist:
            if maximum == minimum:
                a = 0
            else:
                a = (x - minimum) / (maximum - minimum)
            normalized_sublist.append(a)
        
        res.append(normalized_sublist)
    
    return res


def precision(predicted_labels, true_labels):
    return precision_score(true_labels, predicted_labels)

def recall(predicted_labels, true_labels):
    return recall_score(true_labels, predicted_labels)

def f1(predicted_labels, true_labels):
    return f1_score(true_labels, predicted_labels)

def accuracy(predicted_labels, true_labels):
    return accuracy_score(true_labels, predicted_labels)