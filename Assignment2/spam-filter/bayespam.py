import argparse
import os
import math
from enum import Enum

## when a comment shows a number (x), it refers to formula x in the assignment description

class MessageType(Enum):
    REGULAR = 1,
    SPAM = 2


class UseType(Enum):
    TEST = 1
    TRAIN = 2


class Counter():

    def __init__(self):
        self.counter_regular = 0
        self.counter_spam = 0
        self.conditional_log_prob_regular = 0
        self.conditional_log_prob_spam = 0

    def increment_counter(self, message_type):
        """
        Increment a word's frequency count by one, depending on whether it occurred in a regular or spam message.

        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            self.counter_regular += 1
        else:
            self.counter_spam += 1


class Bayespam():
    def clean_up_word(self, token):

        ## Remove punctuation, e.g. '. , : \n ( ) ! ?'
        punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_~\n\t\\x'''
        token = "".join(u for u in token if u not in punctuations)

        ## When the token contains '< > - _ [ ] = @ +' or any number
        ## we assume the token contains no (useful) words
        illegalCharacters = r"<>-_[]=@+0123456789"
        if any(elem in token for elem in illegalCharacters):
            return None

        ## Make lowercase
        token = token.lower()

        ## eliminate when less than 4 letters
        if len(token) < 4:
            return None

        ## return the token
        return token

    def __init__(self):
        self.regular_list = None
        self.spam_list = None
        self.vocab = {}

        ## Counters for the proportion of regular (4) and spam (5) messages
        self.P_regular = 0
        self.P_spam = 0

        ## Counters for the numbers of false/true positives/negatives
        self.confusion_matrix_false_positive = 0
        self.confusion_matrix_false_negative = 0
        self.confusion_matrix_true_positive = 0
        self.confusion_matrix_true_negative = 0

    def list_dirs(self, path):
        """
        Creates a list of both the regular and spam messages in the given file path.

        :param path: File path of the directory containing either the training or test set
        :return: None
        """
        # Check if the directory containing the data exists
        if not os.path.exists(path):
            print("Error: directory %s does not exist." % path)
            exit()

        regular_path = os.path.join(path, 'regular')
        spam_path = os.path.join(path, 'spam')

        # Create a list of the absolute file paths for each regular message
        # Throws an error if no directory named 'regular' exists in the data folder
        try:
            self.regular_list = [os.path.join(
                regular_path, msg) for msg in os.listdir(regular_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'regular'." % path)
            exit()

        # Create a list of the absolute file paths for each spam message
        # Throws an error if no directory named 'spam' exists in the data folder
        try:
            self.spam_list = [os.path.join(spam_path, msg)
                              for msg in os.listdir(spam_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'spam'." % path)
            exit()

    def read_messages(self, message_type, use_type):
        ## This function does part of the training or all of the testing, depending on the variable use_type
        """
        Parse all messages in either the 'regular' or 'spam' directory. Each token is stored in the vocabulary,
        together with a frequency count of its occurrences in both message types.
        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            message_list = self.regular_list
        elif message_type == MessageType.SPAM:
            message_list = self.spam_list
        else:
            message_list = []
            print(
                "Error: input parameter message_type should be MessageType.REGULAR or MessageType.SPAM")
            exit()

        for msg in message_list:
            ## Counters for the proportion of spam (19) / regular (20) messages 
            P_message_spam = self.P_spam
            P_message_regular = self.P_regular
            try:
                # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
                f = open(msg, 'r', encoding='latin1')

                # Loop through each line in the message
                for line in f:
                    # Split the string on the space character, resulting in a list of tokens
                    split_line = line.split(" ")
                    # Loop through the tokens
                    for idx in range(len(split_line)):
                        token = split_line[idx]

                        ## Make sure we count only words, also independent of case
                        ## See function for the full treatment of the words
                        token = self.clean_up_word(token)

                        ## Training in this function means adding words to the vocab
                        if use_type == UseType.TRAIN:
                            if token != None:
                                if token in self.vocab.keys():
                                    ## If the token is already in the vocab, retrieve its counter
                                    counter = self.vocab[token]
                                else:
                                    ## Else: initialize a new counter
                                    counter = Counter()
                                ## Increment the token's counter by one and store in the vocab
                                counter.increment_counter(message_type)
                                self.vocab[token] = counter
                        
                        if use_type == UseType.TEST:
                            ## Check whether the word is in our vocab
                            if token != None and token in self.vocab.keys():
                                ## Get the corresponding counter from the vocab
                                counter = self.vocab[token]
                                ## Sum the probabilities
                                P_message_spam += counter.conditional_log_prob_spam
                                P_message_regular += counter.conditional_log_prob_regular

                if use_type == UseType.TEST:
                    ## Classify as spam or regular
                    msg_spam = True
                    if P_message_regular > P_message_spam:
                        msg_spam = False
                    
                    ## Count true/false positives/negatives
                    if msg_spam == True and message_type == MessageType.SPAM:
                        self.confusion_matrix_true_positive += 1
                    if msg_spam == True and message_type == MessageType.REGULAR:
                        self.confusion_matrix_false_positive += 1
                    if msg_spam == False and message_type == MessageType.SPAM:
                        self.confusion_matrix_false_negative += 1
                    if msg_spam == False and message_type == MessageType.REGULAR:
                        self.confusion_matrix_true_negative += 1

            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()

    def print_vocab(self):
        """
        Print each word in the vocabulary, plus the amount of times it occurs in regular and spam messages.

        :return: None
        """
        for word, counter in self.vocab.items():
            # repr(word) makes sure that special characters such as \t (tab) and \n (newline) are printed.
            print("%s | In regular: %d | In spam: %d" %
                  (repr(word), counter.counter_regular, counter.counter_spam))

    def write_vocab(self, destination_fp, sort_by_freq=True):
        """
        Writes the current vocabulary to a separate .txt file for easier inspection.

        :param destination_fp: Destination file path of the vocabulary file
        :param sort_by_freq: Set to True to sort the vocab by total frequency (descending order)
        :return: None
        """

        if sort_by_freq:
            vocab = sorted(self.vocab.items(
            ), key=lambda x: x[1].counter_regular + x[1].counter_spam, reverse=True)
            vocab = {x[0]: x[1] for x in vocab}
        else:
            vocab = self.vocab

        try:
            f = open(destination_fp, 'w', encoding="latin1")

            for word, counter in vocab.items():
                # repr(word) makes sure that special  characters such as \t (tab) and \n (newline) are printed.
                f.write("%s | In regular: %d | In spam: %d | logProbRegular: %f | logProbSpam: %f\n" % (repr(
                    word), counter.counter_regular, counter.counter_spam, counter.conditional_log_prob_regular, counter.conditional_log_prob_spam))

            f.close()
        except Exception as e:
            print("An error occurred while writing the vocab to a file: ", e)

    def compute_probabilities(self):
        ## (1)
        n_messages_regular = len(self.regular_list)
        ## (2)
        n_messages_spam = len(self.spam_list)
        ## (3)
        n_messages_total = n_messages_regular + n_messages_spam
        ## (4)
        self.P_regular = math.log(float(n_messages_regular)/n_messages_total)
        ## (5)
        self.P_spam = math.log(float(n_messages_spam)/n_messages_total)

        vocab = self.vocab

        ## The rest of the function does what section 2.2 describes
        ## We hope that the code is quite readable due to the variable names
        n_words_regular = 0.0
        n_words_spam = 0.0
        for word, counter in vocab.items():
            n_words_regular += counter.counter_regular
            n_words_spam += counter.counter_spam

        ##A lower epsilon gives better results, but no improvement below 0.3
        epsilon = 0.3

        for word, counter in vocab.items():
            if(counter.counter_regular == 0):
                counter.conditional_log_prob_regular = math.log(
                    epsilon/(n_words_regular+n_words_spam))
            else:
                counter.conditional_log_prob_regular = math.log(
                    float(counter.counter_regular)/n_words_regular)
            if(counter.counter_spam == 0):
                counter.conditional_log_prob_spam = math.log(
                    epsilon/(n_words_regular+n_words_spam))
            else:
                counter.conditional_log_prob_spam = math.log(
                    float(counter.counter_spam)/n_words_spam)

    def print_results(self):
        ## This function prints the confusion matrix and proportions of correct classification
        print("true postives:", self.confusion_matrix_true_positive)
        print("true negatives:", self.confusion_matrix_true_negative)
        print("false postives:", self.confusion_matrix_false_positive)
        print("false negatives:", self.confusion_matrix_false_negative)
        spam_correct = round((self.confusion_matrix_true_positive / (self.confusion_matrix_true_positive + self.confusion_matrix_false_negative)) * 100, 2)
        regular_correct = round((self.confusion_matrix_true_negative / (self.confusion_matrix_true_negative + self.confusion_matrix_false_positive)) * 100, 2)
        print("Spam correctly classified: " + str(spam_correct) + "%")
        print("Regular messages correctly classified: " + str(regular_correct) + "%")



def main():
    # We require the file paths of the training and test sets as input arguments (in that order)
    # The argparse library helps us cleanly parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str,
                        help='File path of the directory containing the training data')
    parser.add_argument('test_path', type=str,
                        help='File path of the directory containing the test data')
    args = parser.parse_args()

    # Read the file path of the folder containing the training set from the input arguments
    train_path = args.train_path
    test_path = args.test_path

    # Initialize a Bayespam object
    bayespam = Bayespam()
    # Initialize a list of the regular and spam message locations in the training folder
    bayespam.list_dirs(train_path)

    # Parse the messages in the regular message directory
    bayespam.read_messages(MessageType.REGULAR, UseType.TRAIN)
    # Parse the messages in the spam message directory
    bayespam.read_messages(MessageType.SPAM, UseType.TRAIN)

    # bayespam.print_vocab()
    bayespam.compute_probabilities()
    bayespam.write_vocab("vocab.txt")

    bayespam.list_dirs(test_path)
    bayespam.read_messages(MessageType.REGULAR, UseType.TEST)
    bayespam.read_messages(MessageType.SPAM, UseType.TEST)

    bayespam.print_results()

    # print("N regular messages: ", len(bayespam.regular_list))
    # print("N spam messages: ", len(bayespam.spam_list))

    """
    Now, implement the follow code yourselves:
    1) A priori class probabilities must be computed from the number of regular and spam messages
    2) The vocabulary must be clean: punctuation and digits must be removed, case insensitive
    3) Conditional probabilities must be computed for every word
    4) Zero probabilities must be replaced by a small estimated value
    5) Bayes rule must be applied on new messages, followed by argmax classification
    6) Errors must be computed on the test set (FAR = false accept rate (misses), FRR = false reject rate (false alarms))
    7) Improve the code and the performance (speed, accuracy)

    Use the same steps to create a class BigramBayespam which implements a classifier using a vocabulary consisting of bigrams
    """


if __name__ == "__main__":
    main()