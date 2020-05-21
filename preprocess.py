import re
import numpy as np
import pandas as pd
import operator

class Preprocess:

    contraction_mapping = {"1950's": "1950s", "1983's": "1983", "ain't": "is not","condition's": "conditions", "aren't": "are not", "Bretzing's": "", "Bundycon's": "Bundycon", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "C'mon": "Come on", "Denzel's": "Denzel", "didn't": "did not",  "doesn't": "does not", "Don't": "Do not", "don't": "do not", "Farmer's": "Farmers", "FBI's": "FBI", "Ferguson's": "Ferguson", "Hammond's": "Hammond", "hadn't": "had not", "hasn't": "has not", "Haven't": "Have not", "haven't": "have not", "he'd": "he would", "Here's": "Here is", "here's": "here is","he'll": "he will", "he's": "he is", "He's": "He is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "I'd": "I had", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "I'm": "I am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "It's": "it is", "Kay's": "Kay", "let's": "let us", "Let's": "let us", "ma'am": "madam", "mayn't": "may not", "Medford's": "Medford", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "Murphy's": "Murphys", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "Paula's": "Paula", "Portland's": "Portlands", "Portlander's": "Portlanders", "publication's": "publications", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "She's": "She is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "Tastebud's": "Tastebuds", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "That's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "There's": "There is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "They're": "They are", "they've": "they have", "to've": "to have", "Trump's": "trump is", "U.S.": "United state", "U.S": "United state", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "We'll": "We will", "we'll've": "we will have", "Wendy's": "Wendy", "we're": "we are", "We're": "We are", "we've": "we have", "We've": "We have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "What's": "What is",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "Who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "Wouldn't": "Would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "You'd": "You had","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "You're": "you are", "you've": "you have", "Zoo's": "zoos", "zoo's": "zoos" }

    mispell_dict = {'&lt;#&gt;': '', 'Ì_': '', ':)': 'midget smiley', 'lor...': 'lor', 'now!': 'now', 'Sorry,': 'Sorry', 'ÌÏ': '', 'å£1000': '', 'URGENT!': 'URGENT', 'you!': 'you', 'you?': 'you ?', 'now?': 'now ?', '4*': '', '&lt;DECIMAL&gt;': '', 'Ok...': 'Ok', ':-)': 'basic smiley', 'me?': 'me ?', 'å£100': '', 'liao...': 'liao', 'å£2000': '', '86688': '', 'å£5000': '', 'T&Cs': 'Terms and Conditions', 'u?': 'you ?', 'I.ll': 'I will', 'tonight?':'tonight', 'lar...': 'lar', 'GUARANTEED.': 'Guaranteed', 'No:': 'No', 'landline.': 'landline', 'princess!': 'princess', 'right?':'right', 'PRIVATE!': 'private', 'NOW!': 'now', 'today?': 'today', ':-(': 'boo hoo', 'Aight,': 'Aight', 'MobileUpd8': 'Mobile update', 'lei...': 'lei', 'already?': 'already', "How's": 'how is', 'Haha...': 'Haha', 'it?': 'it', 'ah?': 'ah', 'week!': 'week', 'me...': 'me', 'prize!': 'prize', 'FREE!': 'Free', 'it!': 'it', 'babe,': 'babe', 'Yeah,': 'Yeah', 'YOU!': 'YOU', 'T&C': 'Terms and condition', 'it...': 'it', 'leh...':'leh', 'wat...': 'wat', 'easy,':'easy', 'un-redeemed': 'unredeemed', 'kiss*': 'kiss', 'Sir,':'Sir', 'Urgent!': 'Urgent', 'already...': 'already', '*grins*': 'grins', 'u...': 'you', "''": '', 'smileyi': 'smiley', 'now...': 'now', 'me..': 'me', 'it..': 'it', 'doing?': 'doing', 'mobile!': 'mobile', 'u!': 'u', 'out!': 'out', 'night?': 'night', 'today!': 'today', 'there?': 'there', 'rite...': 'rite', ':(': 'sad turtle smiley', 'ok,': 'ok', 'websitezed.co.uk': 'website', 'word:': 'word', 'up?': 'up', "condition's": "conditions", 'Ts&Cs': 'Terms and conditions', 'Hee...': 'Hee', 'ok?': 'ok', 'there': 'there', "Joy's": "Joys", 'good,': 'good', 'baby!': 'baby', 'day?': 'day', 'oredi...': 'oredi', 'Yes,': 'Yes', 'simple..': 'simple', 'wat?': 'wat', 'tomorrow,': 'tomorrow', 'later?': 'later', 'Cool,': 'Cool', 'afternoon,': 'afternoon', 'Ltd,': 'Ltd','Haha,': 'Haha', 'tmr?': 'tmr', 'Awesome,': 'Awesome', 'Hi,':'Hi', 'yet?': 'yet', 'Congrats!': 'Congrats', 'special-call': 'special call', 'R*reveal': 'reveal' ,'U-find': 'U find', 'smileybasic': 'smiley basic', 'again!': 'again', 'ok...': 'ok', 'ah...': 'ah', 'Yo': 'Yo', 'ni8': 'night', '&lt;TIME&gt;': '', 'not,': 'not', 'lor,': 'lor', 'going?': 'going', 'week?': 'week', 'night,': 'night', 'back?': 'back', 'Hello,': 'Hello', 'that!': 'that' }

    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    #initializer
    def __init__(self, text):
        self.text = text;

    #change the link to "website"
    def clean_text(self,texts):
        texts = texts.replace('\n', ' ')
        if 'www.' in texts or 'http:' in texts or 'https:' in texts or '.com' in texts:
            texts = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "website", texts)
        return texts

    #contraction  mapping
    def clean_contractions(self,text, mapping):
        specials = ["’", "‘", "´", "`"]
        for s in specials:
            text = text.replace(s, "'")
        text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
        return text

    def correct_spelling(self,x, dic):
        for word in dic.keys():
            x = x.replace(word, dic[word])
        return x

    def clean_special_chars(self,text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p])

        specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
        for s in specials:
            text = text.replace(s, specials[s])

        return text
