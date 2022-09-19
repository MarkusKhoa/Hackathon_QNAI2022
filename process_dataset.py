from base64 import encode
from tqdm import tqdm
import os
import pandas as pd
import re
import demoji
from nltk.stem.porter import PorterStemmer
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("./VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

dataset = './dataset/crawl_final.csv'
df = pd.read_csv(dataset)
stop_word = []
txt_file = open("./dataset/vietnamese-stopwords-dash.txt", "r", encoding = "utf-8")
file_content = txt_file.read()
content_list = file_content.split("\n")
stemmer = PorterStemmer()
SENTI_DICT = {
    1: "very_negative",
    2: "negative",
    3: "normal",
    4: "positive",
    5: "very_positive"
}

def remove_url(text):
    text = re.sub(r"http\S+", "", text)
    return text

def handle_emoji(string):
    emojis = demoji.findall(string)

    for emoji in emojis:
        string = string.replace(emoji, " " + emojis[emoji].split(":")[0])

    return string

def remove_stopwords(text):
    text = [word for word in text if word not in content_list]
    new_text = " ".join(text)
    # return text
    return new_text

def stemming(text):
    text = [stemmer.stem(word) for word in text]
    new_text = " ".join(text)
    # return text
    return new_text

def word_tokenizer(text):
    tokens = rdrsegmenter.tokenize(text)
    tokens = [t for ts in tokens for t in ts]
    word_segmented_text = " ".join(tokens)
    return word_segmented_text
    # return tokens

def preprocessing(text):
    text = remove_url(text)
    text = handle_emoji(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'(ks)', 'khách_sạn', text)
    # text = re.sub(r'(ko)', 'không', text)
    text = word_tokenizer(text)
    # text = remove_stopwords(text)
    # text = " ".join(text[0])
    return text

def process_labels_csv(data_frame):
    labels_lst = []
    for i in range(len(data_frame)):
        sample_labels = []
        for col in data_frame.columns.values[2:8]:
            if data_frame[col][i] != 0:
                polarity = data_frame[col][i]
                label = f"{col}#{SENTI_DICT[polarity]}"
                sample_labels.append(f"{{{label}}}")
                label_str = ", ".join([ele for ele in sample_labels])
        
        labels_lst.append(label_str)
    data_frame["label"] = labels_lst
    return data_frame

def process():
    df1 = df.copy()
    tqdm.pandas()
    df1['clean_review'] = df1['Review'].progress_map(preprocessing)
    process_labels_csv(df1)
    df1.to_csv('./dataset/clean_crawl_final.csv', encoding='utf-8-sig')

if __name__ == "__main__":
    process()
    print("Everything is done!")