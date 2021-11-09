import html
import re
import string
import unicodedata
import nltk
import benepar
import os 

import pandas as pd
import numpy as np

from collections import Counter
from tqdm.auto import tqdm
from textblob import TextBlob

def clean_base(t):
    """Clean the text in argument:
       
    :param t (string): text to clean
    
    :return (string): text t cleaned
    """
    if pd.notna(t):
        t = t.encode('ascii', errors='ignore').decode('unicode_escape', errors='ignore')
        t = html.unescape(t)
        t = unicodedata.normalize('NFKD', t)
        t = re.sub(r'http\S+', '', t)
        # remove hashtags (FND)
        t = re.sub(r'#\S+', '', t)
        # remove pic twitter (FND)
        t = re.sub(r'pic.twitter.com\S*', '', t)
        t = re.sub(r'[\w\.-]+@[\w\.-]+', '', t)
        html_tags = re.compile(r'<[^>]+>')
        t = html_tags.sub('', t)
        #t = re.sub(r'[0-9]', '', t) #delete digit
        #t = t.strip(string.punctuation + ' ') 
        #may remove . at the encd of content but also ] and for authors
    return t

def clean_content(t):
    # remove all special characters but ,.!?
    t = re.sub(r'["#$%&()*+\-/:;<=>@\[\\\]^_`{|}~]', ' ', str(t))
    # remove big spaces
    t = re.sub(r'\s+', ' ', t)
        # remove series of puncutations, space before punctuations and add one space after punctuations
    t = re.sub(r'\s?([,!.?])([ ,!.?]*)', r'\1 ', t)
    # old versions to keep track
    #t = re.sub(r'[ !.?]{2,}', '. ', t) 
    #t = re.sub(r'([,.!?])(\S)', r'\1 \2', t)
    return t.strip(' ,')
    
def clean_fake(serie):
    return serie.str.replace(r'\s*story continued below\s*', ' ', case=False)

def length_encode(sent_table, tokenizer):
    """Encode each sentences of the articles and return their lengths
    
    :param sent_table (list(string)): list of one article sentences
    :param tokenizer: tokenizer object of Huggingface library
    
    :return (list(int)): list of nb of token for each sentence
    """
    sent_tokens = tokenizer(sent_table, 
                            add_special_tokens=False, 
                            return_attention_mask=False, 
                            return_token_type_ids=False, 
                            return_length=True, 
                            verbose=False)
    
    return sent_tokens["length"]


def verification_sent(sent_sizes):
    """ Verify if each sentence as less than "nb_tokens" tokens
    
    :param sent_size (list(int)): list of nb of tokens for each sentence
    
    :return (bool): True if at least one sentence is bigger than nb_tokens, False otherwise
    """
    for sent_size in sent_sizes:
        if sent_size > 512:
            return True
    return False

def sent_mask_trunc(t, nb_tokens=512, head=128):
    """Compute mask to us the begining and the end of the article as input for BERT
    
    :param t (list(int)): list of nb of tokens for each sentence
    :param nb_tokens (int): nb of tokens we want to use as input for Bert
    :param head (int): nb of tokens we want from the begining of the article (nb_tokens - head for the end)
    
    :return (list(int)): list of integer that represente the groups appartenance for each sentence.
    """
    
    t = t.sent_size
    
    nb_tokens_special = nb_tokens - 2
    
    if head > nb_tokens_special:
        raise ValueError("head must be below {}".format(nb_tokens_special))
   
    mask = []
    previous_size = 0
    real_head = 0
    s_nb = 0
        
    for size in t:
        current_size = previous_size + size
        
        if (current_size > head) and (s_nb == 0):
    
            prev_diff = head - previous_size
            current_diff = current_size - head 
            
            if (current_diff < prev_diff) and (current_size <= nb_tokens_special):
                mask.append(s_nb)
                s_nb += 1
                real_head = current_size
            else:
                s_nb += 1
                mask.append(s_nb)
                real_head = previous_size
        
        else:
            mask.append(s_nb)
            
        previous_size = current_size
    
    tail = nb_tokens_special - real_head
    i = -1
    current_size = t[i]
    
    #print("mask", mask)
    #print("i", i)
    #print("mask[i]", mask[i])
    #print("t[i]", t[i])
    #print("current_size", current_size)
    
    while (current_size < tail) and (mask[i] == 1): 
        mask[i] = 0
        i -= 1
        current_size += t[i]
        
        #print("mask", mask)
        #print("i", i)
        #print("mask[i]", mask[i])
        #print("t[i]", t[i])  
        #print("current_size", current_size)
    
    return np.array(mask)

def increase_data_split(row_df):
    """ Create new articles from the sentences and masks of the original article of the parameters
    
    :param row_df (pd.Serie): row with the article content to extend
    
    :return (pd.DataFrame): new DF, each row is a part of te original content
    """
    columns_to_drop = ["sentences", "sent_size", "verification", "sent_mask"]
    
    to_concat = []
    
    for i in range(row_df.sent_mask[-1] + 1):
        inds = np.where(row_df.sent_mask == i)[0]
        
        if len(inds) != 0:
            row = row_df.copy()
            #row["short_content"] = " ".join(row_df.sentences[inds[0]:inds[-1]+1])
            row["old_content"] = row["content"]
            row["content"] = " ".join(np.array(row_df.sentences)[inds])  
            row["sent_id"] = i
            
            #row["nb_tokens"] = sum(row.sent_size[inds[0]:inds[-1]+1])
            row["nb_tokens"] = sum(np.array(row.sent_size)[inds])
            #row["tokens"] = [t for tokens in row.sent_tokens[inds[0]:inds[-1]+1] for t in tokens]
            #row["tokens"].insert(0,101)
            #row["tokens"].append(102)
            
            to_concat.append(pd.DataFrame([row.drop(columns_to_drop)]))
        
    return pd.concat(to_concat)
    
def extend_data(data, tokenizer, nb_tokens=512, mask_f=sent_mask_trunc, arg_mask=True, path_save=None):
    """Increase data by splitting articles by group of max "nb_tokens" (cut at sentences)
    
    :param data (pd.DataFrame): df to increase
    :param tokenizer: tokenizer object of Huggingface library
    :param nb_tokens (int): nb of tokens as input for the Bert model
    :param mask_f (function): strategy to create mask (extend or truncation) if None no strategy
    :param arg_mask (bool or int): either to use a limit for mask strategy or the size of the head if truncation
    
    :return (pd.DataFrame): df extended
    """
    df = data.copy()
    
    if mask_f is not None:
        
        df["sentences"] = df.content.map(lambda c: nltk.tokenize.sent_tokenize(c))

        #df["sent_tokens"] = df.sentences.map(lambda s: encode_tables_loop(s, tokenizer))
        #df["sent_size"] = df.sent_tokens.map(lambda tokens: [len(t) for t in tokens])
        print("1")
        
        df = df[df["sentences"].map(len) != 0]
        
        df["sent_size"] = df.sentences.map(lambda s: length_encode(s, tokenizer))
        
        df["verification"] = df.sent_size.map(lambda ss: verification_sent(ss))
        df = df[df.verification == False]
        
        df["sent_mask"] = df.apply(lambda ss: mask_f(ss, nb_tokens, arg_mask), axis=1)

        extended_df = pd.concat(df.apply(lambda row: increase_data_split(row), axis=1).tolist())
        
        if extended_df.sent_id.nunique() == 1:
            extended_df = extended_df.drop(["sent_id"], axis=1)
        else:
            extended_df = extended_df.reset_index().rename(columns={'index': 'full_article_index'})
                                                           
    else:
        df["nb_tokens"] = df.content.map(lambda s: length_encode(s, tokenizer))
        extended_df = df.copy()
        
    if path_save is not None:
        extended_df.to_pickle(path_save)
        
    return extended_df

def encode(instance, tokenizer, nb_tokens):
    """Encode the content of the article thanks to the tokenizer
    
    :param instance (string or list(string)): content to encode
    :param tokenizer: tokenizer object of Huggingface library
    :param nb_tokens (int): nb max of tokens
    
    :return (list(int)): list of tokens of the instance encoded by the tokenizer
    """
    features = tokenizer(instance,  
                         add_special_tokens=True,
                         max_length=nb_tokens,
                         truncation=True,
                         padding='max_length', 
                         return_attention_mask=True,
                         return_token_type_ids=False)
                         #return_tensors="tf")
    return features
    
def encode_plus(df, tokenizer, nb_tokens, path_save=None):
    """Encode, complete df + return data accepted by model
    
    :param df (pd.DataFrame): df with content to encode
    :param tokenizer : tokenizer object of Huggingface library
    :param nb_tokens (int): nb max of tokens
    
    : return (list(np.array(list(in))): input_ids and attention_mask of all data
    """
    encoding = encode(df.content.values.tolist(), tokenizer, nb_tokens)
    df["input_ids"] = list(encoding["input_ids"])
    df["attention_mask"] = list(encoding["attention_mask"])
    
    if path_save is not None:
        df.to_pickle(path_save)
        
    return [np.array(df.input_ids.values.tolist()), np.array(df.attention_mask.values.tolist())], df

def production_except_remove(list_feat):
    reg_sub = re.compile(r"\[|\]|<", re.IGNORECASE)
    
    final_lf = [reg_sub.sub('', feat) for feat in list_feat]

    return final_lf  
    
def cfg_features(text, nlp, parser, path_models):

    doc = nlp(text)
    prods = []
    limit_sent = 290
    for sent in doc.sents:
        # limitation from benepar (raise an error)
        sl = len(sent)
        if sl <= limit_sent :
            prods += parser.parse(sent.text.lower()).productions()
        # separate the sentence in blocs of 300 words and the rest for the last bloc
        # do this or just take the first 300 words or pass this sentence
        else:
            p = 0
            for b in ([limit_sent] * int(sl / limit_sent)) + [sl % limit_sent]:
                prods += parser.parse(sent[p: p + b].text.lower()).productions()
                p += b
    return prods

def cfg_filter(rules):
    lexical, non_lexical = [], []
    for rule in rules:
        if rule.is_lexical():
            lexical.append(rule)
        elif rule.is_nonlexical():
            non_lexical.append(rule)
    return {"lexical": lexical, "non_lexical": non_lexical}

def prod_to_str(dic):
    new_dic = {}
    for k,l in dic.items():
        new_dic[k] = [str(p) for p in l]
    return new_dic
    
def cfg_feat(df, nlp, path_models, path_save=None, ind=""):
    tqdm.pandas()
    
    fnd_df = df.copy()
    
    # load Berkeley Neural Parser based on
    # Constituency Parsing with a Self-Attentive Encoder.
    parser = benepar.Parser(os.path.join(path_models, "benepar_en2"))
    
    print("Context Free Grammar")
    fnd_df["text_cfg"] = fnd_df.text.progress_map(lambda t: cfg_features(t, nlp, parser, path_models))
    
    print("CFG separation lexical")
    fnd_df["text_cfg_sep"] = fnd_df.text_cfg.progress_map(cfg_filter)
    
    print("CFG string representations")
    fnd_df["text_cfg_sep_str"] = fnd_df.text_cfg_sep.progress_map(prod_to_str)
    
    if path_save is not None:
        fnd_df.to_pickle(path_save)
        
    return fnd_df

def readability_features(text, nlp):
    doc = nlp(text)
    read_scores = {
        "fkgl": np.round(doc._.flesch_kincaid_grade_level, 2),
        "fkre": np.round(doc._.flesch_kincaid_reading_ease, 2),
        "dc": np.round(doc._.dale_chall, 2),
        "smog": np.round(doc._.smog, 2),
        "cli": np.round(doc._.coleman_liau_index, 2),
        "ari": np.round(doc._.automated_readability_index, 2),
        "fc": np.round(doc._.forcast, 2)
    }
    return read_scores


def unique_features(text_split, once=True):
    """
    Function that count the proportion of unique words or tags.

    Caution: there are different ways to understand unique:
     - the number of different words
    - the number of words that appears only once
    Both interpretation can be useful.
    """

    if once:
        # exist better way ?
        uw = len([v for v in Counter(text_split).values() if v == 1])
    else:
        uw = len(set(text_split))
    return uw


def diversity_words(text):
    text_split = text.split()
    tot = len(text_split)

    diff_words = unique_features(text_split, False) / tot
    uni_words = unique_features(text_split, True) / tot

    return {"diff_words": np.round(diff_words, 2), "uni_words": np.round(uni_words, 2)}


def verify_div(a, b):
    # In the cases we use this division if a == 0 then b == 0
    # but we want to prevent division by 0
    if a == 0:
        return 0
    else:
        return a / b


def diversity_pos(text_tup):
    """
    Same things for tags but the proportion can also change.
    Example :

    proportion of unique NOUN =
    - number of Noun that appears only once / nb total of NOUN
    - number of Noun that appears only once / nb total of words
    - DEFAULT : number of different NOUN / nb total of NOUN
    - number of different NOUN / nb total of words
    """

    all_noun, all_verb, all_adj, all_adv = [], [], [], []

    for t, l, p, _, _ in text_tup:
        if (p == "NOUN") or (p == "PROPN"):
            all_noun.append((l, p))
        elif p == "VERB":
            all_verb.append((l, p))
        elif p == "ADJ":
            all_adj.append((l, p))
        elif p == "ADV":
            all_adv.append((l, p))

    diff_noun = unique_features(all_noun, False)
    uni_noun = unique_features(all_noun, True)

    diff_verb = unique_features(all_verb, False)
    uni_verb = unique_features(all_verb, True)

    diff_adj = unique_features(all_adj, False)
    uni_adj = unique_features(all_adj, True)

    diff_adv = unique_features(all_adv, False)
    uni_adv = unique_features(all_adv, True)

    diversity_pos = {"diff_noun_n": np.round(verify_div(diff_noun, len(all_noun)), 2),
                     "diff_noun_w": np.round(diff_noun / len(text_tup), 2),
                     "uni_noun_n": np.round(verify_div(uni_noun, len(all_noun)), 2),
                     "uni_noun_w": np.round(uni_noun / len(text_tup), 2),
                     "diff_verb_n": np.round(verify_div(diff_verb, len(all_verb)), 2),
                     "diff_verb_w": np.round(diff_verb / len(text_tup), 2),
                     "uni_verb_n": np.round(verify_div(uni_verb, len(all_verb)), 2),
                     "uni_verb_w": np.round(uni_verb / len(text_tup), 2),
                     "diff_adj_n": np.round(verify_div(diff_adj, len(all_adj)), 2),
                     "diff_adj_w": np.round(diff_adj / len(text_tup), 2),
                     "uni_adj_n": np.round(verify_div(uni_adj, len(all_adj)), 2),
                     "uni_adj_w": np.round(uni_adj / len(text_tup), 2),
                     "diff_adv_n": np.round(verify_div(diff_adv, len(all_adv)), 2),
                     "diff_adv_w": np.round(diff_adv / len(text_tup), 2),
                     "uni_adv_n": np.round(verify_div(uni_adv, len(all_adv)), 2),
                     "uni_adv_w": np.round(uni_adv / len(text_tup), 2)}

    return diversity_pos


def subjectivity_features(text, nlp, path_models):
    path_bias = os.path.join(path_models, "bias_related_lexicons/")

    bias_lexicon = [line.rstrip("\n") for line in open(path_bias + "bias-lexicon.txt", encoding='utf-8-sig')]
    factive_lexicon = [line.rstrip("\n") for line in open(path_bias + "factives_hooper1975.txt", encoding='utf-8-sig')]
    report_lexicon = [line.rstrip("\n") for line in open(path_bias + "report_verbs.txt", encoding='utf-8-sig')]

    # implicative_lexicon = [line.rstrip("\n") for line in open(path_bias + "implicatives_karttunen1971.txt", encoding='utf-8-sig')]
    # assertives_lexicon = [line.rstrip("\n") for line in open(path_bias + "assertives_hooper1975.txt", encoding='utf-8-sig')]
    # hedges_lexicon = [line.rstrip("\n") for line in open(path_bias + "hedges_hyland2005.txt", encoding='utf-8-sig')]

    doc = nlp(text)
    nb_words = len(TextBlob(text).words)
    subj = {"bias": 0, "factive": 0, "report": 0}
    for t in doc:
        if t.lemma_ in bias_lexicon:
            subj["bias"] += 1
        elif t.lemma_ in factive_lexicon:
            subj["factive"] += 1
        elif t.lemma_ in report_lexicon:
            subj["report"] += 1
    return {k: v / nb_words for k, v in subj.items()}


def sentiment_features(text):
    sent = TextBlob(text).sentiment
    sent_dic = {"polarity": np.round(sent.polarity, 2),
                "subjectivity": np.round(sent.subjectivity, 2)}
    return sent_dic


def quantity_features(text):
    blob = TextBlob(text)

    char_word = [len(w) for w in blob.words]
    word_sent = [len(s.words) for s in blob.sentences]
    sent_para = [len(TextBlob(p).sentences) for p in text.split("\n") if len(p) != 0]

    nb_char = sum(char_word)
    nb_word = len(char_word)
    nb_sent = len(word_sent)
    nb_para = len(sent_para)

    avg_char_word = np.round(np.mean(char_word), 1)
    avg_word_sent = np.round(np.mean(word_sent), 1)
    avg_sent_para = np.round(np.mean(sent_para), 1)

    quant_dict = {"nb_char": nb_char,
                  "nb_word": nb_word,
                  "nb_sent": nb_sent,
                  "nb_para": nb_para,
                  "avg_char_word": avg_char_word,
                  "avg_word_sent": avg_word_sent,
                  "avg_sent_para": avg_sent_para}

    return quant_dict

def punctuation_features(text, prop=True):
    nb_words = len(TextBlob(text).words)
    punct = ["!", "?", "...", ",", ";", ":"]
    count_punct = {p:text.count(p) for p in punct}
    count_punct["total"] = sum(count_punct.values())
    if prop:
        count_punct = {k: np.round(v/nb_words * 100, 2) for k,v in count_punct.items()}
    return count_punct

def empath_features(text, lexicon, normalize=False):
    informal_cat = ["swearing_terms", "swear", "filler", "netspeak", "assent", "nonflu"]

    sentiment_pos_cat = ["positive_emotion", "cheerfulness", "love", "trust", "affection", "joy"]
    sentiment_pos_bonus_cat = ["attractive", "optimism", "healing", "sympathy", "valuable", "fun", "contentment", "beauty"]

    sentiment_neg_cat = ["negative_emotion", "hate", "fear", "suffering", "nervousness", "irritability", "disgust", "sadness", "anger", "disappointment", "exasperation", "deception"]
    sentiment_neg_bonus_cat = ["pain", "torment", "envy", "rage", "injury", "crime", "dispute", "weakness", "kill", "aggression", "death", "violence", "neglect", "fight", "shame"]

    cogproc_cat = ["cogproc", "insight", "cause", "discrep", "tentat", "certain"]
    cogproc_bonus_cat = ["anticipation", "negotiate"]

    percept_cat = ["percept", "see", "hear", "feel", "hearing", "listen", "emotional"]
    percept_bonus_cat = ["smell", "warmth", "lust", "surprise", "confusion"]
    
    all_cat = informal_cat + sentiment_pos_cat + sentiment_pos_bonus_cat + sentiment_neg_cat + sentiment_neg_bonus_cat + cogproc_cat + cogproc_bonus_cat + percept_cat + percept_bonus_cat
    
    empath_dict = lexicon.analyze(text, categories=all_cat, normalize=normalize)
    
    return empath_dict

def all_pos_features(text, nlp):
    doc = nlp(text)
    return [(t.text, t.lemma_, t.pos_, t.tag_, t.dep_) for t in doc]


def pos_tags_feat(fnd_df, nlp, path_save=None):
    tqdm.pandas()
    
    for i,syn in enumerate(["pos", "tag", "dep"]):
        print("Extract : {}".format(syn))
        fnd_df["text_" + syn] = fnd_df.text_tpos.progress_map(lambda a_tup: " ".join([t[i+2] for t in a_tup]))
        
    if path_save is not None:
        fnd_df.to_pickle(path_save)
    
    return fnd_df
    
def cat_feat(fnd_df, nlp, lexicon, path_models, path_save=None):
    tqdm.pandas()
    
    print("Readability score")
    fnd_df["text_read"] = fnd_df.text.progress_map(lambda t: readability_features(t, nlp))
    
    print("Empath")
    fnd_df["text_empath"] = fnd_df.text.progress_map(lambda t: empath_features(t, lexicon, normalize=True))
    
    print("Diversity of words")
    fnd_df["text_div_words"] = fnd_df.text.progress_map(diversity_words)
    
    print("All pos tags")
    fnd_df["text_tpos"] = fnd_df.text.progress_map(lambda t: all_pos_features(t, nlp))
    
    print("Diversity of tags")
    fnd_df["text_div_pos"] = fnd_df.text_tpos.progress_map(diversity_pos)
    
    print("Subjectivity")
    fnd_df["text_subj"] = fnd_df.text.progress_map(lambda t: subjectivity_features(t, nlp, path_models))
    
    print("Sentiment")
    fnd_df["text_sent"] = fnd_df.text.progress_map(sentiment_features)
    
    print("Quantity")
    fnd_df["text_quant"] = fnd_df.text.progress_map(quantity_features)
    
    print("Punctuation")
    fnd_df["text_punct"] = fnd_df.text.progress_map(punctuation_features)
    
    if path_save is not None:
        fnd_df.to_pickle(path_save)

    return fnd_df