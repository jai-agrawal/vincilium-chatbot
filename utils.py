import json
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
import keras
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from read import postgresql_to_dataframe
import datetime as dt
from collections import Counter
import pandas as pd

def get_date_range():
    """
    Get date range.
    :return:
    """
    try:
        print("Please enter the start date (YYYY-MM-DD): ")
        start_date = input()
        print("Please enter the end date (YYYY-MM-DD): ")
        end_date = input()
        return start_date, end_date
    except:
        print("Invalid date format. Please try again.")
        return get_date_range()

def append_to_json(filename, key, value):
    """
    Append entry to json file.
    """
    with open(filename, 'r+') as f:
        data = json.load(f)
        data[key] = value
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()

def show_all(source_or_recon):
    """
    Show all the sources or recons.
    """
    print(f"Here are the {source_or_recon}s:")
    recons = postgresql_to_dataframe(f'st_{source_or_recon}_config')[f'{source_or_recon}_name'].tolist()
    for i, recon in enumerate(recons):
        print(f"{i + 1}. {recon}")

def access_recon(recon_name, start_date, end_date):
    try:
        df = postgresql_to_dataframe(f"br_rec_{recon_name.lower()}")
        df_filtered = df.loc[df['created_date'].between(start_date, end_date, inclusive=False)]
        count = len(df_filtered)
        match_statuses = list(df_filtered['match_status'])
        counts = Counter(match_statuses)
        matched = counts['MATCHED']
        proposed_match = counts['PROPOSED MATCH']
        unmatched = count - (matched - proposed_match)
        print(f"It looks like {count} transactions were added today, {matched} are in status MATCHED,"
              f" {proposed_match} are in status PROPOSED_MATCH "
              f"and {unmatched} are unmatched.")
    except:
        print("Sorry, I don't understand. Let's start again.")
        inp = input()
        chat(inp)

def similar(sentence1, sentence2):
    """
    Get similarity between two sentences.
    """
    nlp = spacy.load('en_core_web_sm')
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)
    return doc1.similarity(doc2)


def error_handler(tag, inp):
    """
    Handle errors.
    """
    print("Is that correct?\nPress 'y' for yes and 'n' for no.")
    yn_inp = input()
    try:
        if yn_inp.lower() == 'y':
            return tag
        elif yn_inp.lower() == 'n':
            print("Okay. I can perform the following tasks:")
            print("\t1. Show all the recons")
            print("\t2. Show all the sources")
            print("\t3. Access a recon")
            print("Please enter the number of the task you want to perform.")
            num_inp = input()
            if num_inp == '1':
                # add this to the json
                append_to_json('saved_responses.json', inp, 'recon')
                tag = 'recon'
                return tag
            elif num_inp == '2':
                # add this to the json
                append_to_json('saved_responses.json', inp, 'source')
                tag = 'source'
                return tag
            elif num_inp == '3':
                # add this to the json
                append_to_json('saved_responses.json', inp, 'recon_name')
                tag = 'recon_name'
                return tag
    except:
        print("Sorry, I don't understand. Let's start again.")
        inp = input()
        chat(inp)


def tag_handler(tag, inp, saved):
    """
    Handle tags.
    """
    repeat_task = ''
    task = False
    with open("intents.json") as file:
        data = json.load(file)

    for i in data['intents']:
        if i['tag'] == tag:
            print(np.random.choice(i['responses']))
            if i['type'] == 'task':
                task = True

    if task and not saved:
        tag = error_handler(tag, inp)

    if tag == 'recon' or tag == 'source':
        show_all(tag[0])
        repeat_task = 'else '

    elif tag == 'recon_name':
        # check if the recon exists
        start_date, end_date = get_date_range()
        recon_name = input("Please enter the name of your recon: ")
        access_recon(recon_name, start_date, end_date)
        repeat_task = 'else '

    elif tag == 'goodbye':
        quit()

    print(f"What {repeat_task}can I help you with?")
    inp = input()
    chat(inp)


def chat(inp):
    """
    Chat with user.
    """
    with open("saved_responses.json") as file:
        data = json.load(file)
    saved = False
    # load trained model
    model = keras.models.load_model('model/chat_model.h5')
    # load tokenizer object
    with open('model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    # load label encoder object
    with open('model/label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    while True:
        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        # check for prev cases
        for key in data:
            similarity_index = similar(inp, key)
            if similarity_index > 0.8:
                tag = data[key]
                saved = True
        tag_handler(tag, inp, saved)