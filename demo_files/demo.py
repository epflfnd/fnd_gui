import shap
import re
import spacy
import os
import helper

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import custom_transformers as ct
import tensorflow as tf
import transformers as hf

from empath import Empath
from matplotlib import cm, colors
from joblib import load
from PIL import Image
from nltk.tokenize import sent_tokenize
from spacy_readability import Readability
#st.caching.clear_cache()

@st.cache(allow_output_mutation=True)
def load_data():
    cfg_tags = Image.open("code/cfg_tags.jpg")
    
    path_data = "data/fnd/dataframe/demo/"

    demo_df = pd.read_pickle(path_data + "demo_df.pkl").reset_index()
    demo_lex_df = pd.read_pickle(path_data + "demo_lex_df.pkl")
    demo_cfg_df = pd.read_pickle(path_data + "demo_cfg_df.pkl")
    demo_sem_df = pd.read_pickle(path_data + "demo_sem_df.pkl")
    demo_bert_df = pd.read_pickle(path_data + "demo_bert_df.pkl")
    demo_ml_df = pd.read_pickle(path_data + "demo_ml_df.pkl")
    demo_ml_wb_df = pd.read_pickle(path_data + "demo_ml_wb_df.pkl")
    
    models = [('lex_model.joblib', demo_lex_df), 
              ('cfg_model.joblib', demo_cfg_df), 
              ('sem_model.joblib', demo_sem_df), 
              #('ml_model.joblib', demo_ml_df)]
              ('ml_wb_model.joblib', demo_ml_wb_df)]
    
    #with tf.device('/device:GPU:1'):
        #bert_model = tf.keras.models.load_model(path + 'bert_model_h5')
    
    mod_exp_shap = []
    
    for mod, df in models:
        model = load("model/fnd/demo/" + mod)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        mod_exp_shap.append((model, explainer, shap_values, df))
    
    mod_exp_shap.append(demo_bert_df)
    
    return demo_df, mod_exp_shap, cfg_tags

demo_df, mod_exp_shap, cfg_tags = load_data()

def token_vect(doc):
    return doc

def sum_score(X, y):
    return np.squeeze(np.asarray(X.sum(axis=0)))
    
@st.cache(allow_output_mutation=True)
def load_data_clean():
    path_df = "data/fnd/dataframe/pol_clean_results" 
    path_learners = "model/fnd/pol_clean_results"
    path_nlp_models = "model/nlp_model"
    
    empath = load(os.path.join(path_nlp_models, "empath/lexicon_custom"))
    nlp = spacy.load(os.path.join(path_nlp_models, "en_core_web_sm-2.3.1"))
    nlp.add_pipe(Readability(), last=True)
    
    train_feat_df = pd.read_pickle(os.path.join(path_df, "train_feat_df.pkl"))
    
    lex_pipe = load(os.path.join(path_learners, "lex_pipe.pkl"))
    cfg_pipe = load(os.path.join(path_learners, "cfg_pipe_non_lexical.pkl"))
    cat_pipe = load(os.path.join(path_learners, "cat_empath_pipe.pkl"))
    
    pipes = {"lex":lex_pipe, "cfg":cfg_pipe, "cat":cat_pipe}
    
    lex_model = load(os.path.join(path_learners, "lex_model.joblib"))
    cfg_model = load(os.path.join(path_learners, "cfg_model_non_lexical.joblib"))
    cat_model = load(os.path.join(path_learners, "cat_empath_model.joblib"))
    bert_model = tf.keras.models.load_model(os.path.join(path_learners, 'bert_model_h5'))
    meta_learner = load(os.path.join(path_learners, "meta_learner_empath_nn_lexical.joblib"))
    
    models = {"lex":lex_model, "cfg":cfg_model, "cat":cat_model, "bert":bert_model, "meta_learner": meta_learner}
    
    return nlp, empath, pipes, models

nlp, empath, pipes, models = load_data_clean()

ANALYZE = {
    "Classification": mod_exp_shap[3],
    "Lexical": mod_exp_shap[0], 
    "Grammatical": mod_exp_shap[1], 
    "Categorized": mod_exp_shap[2],
    "BERT":mod_exp_shap[4]
}

names = {"lex": "Lexical", "cfg": "Grammatical", "sem": "Categorized", "Bert": "BERT"}
sem_dict = {'informal': 'informal language', 'swear': 'swear words', 'netspeak': 'netspeak', 'assent': 'assent', 'nonflu': 'nonfluencies', 'filler': 'fillers', 'diff_words': 'prop different words', 'uni_words': 'prop unique words', 'diff_noun_n': 'prop different noun (/noun)', 'diff_noun_w': 'prop different noun (/word)', 'uni_noun_n': 'prop unique noun (/noun)', 'uni_noun_w': 'prop unique noun (/word)', 'diff_verb_n': 'prop different verb (/verb)', 'diff_verb_w': 'prop different verb (/word)', 'uni_verb_n': 'prop unique verb (/verb)', 'uni_verb_w': 'prop unique verb (/word)', 'diff_adj_n': 'prop different adjective (/adj)', 'diff_adj_w': 'prop different adjective (/word)', 'uni_adj_n': 'prop unique adjective (/adj)', 'uni_adj_w': 'prop unique adjective (/word)', 'diff_adv_n': 'prop different adverb (/adv)', 'diff_adv_w': 'prop different adverb (/word)', 'uni_adv_n': 'prop unique adverb (/adv)', 'uni_adv_w': 'prop unique adverb (/adv)', 'bias': 'biased terms', 'factive': 'factive verbs', 'report': 'report verbs', 'affect': 'affective processes', 'posemo': 'positive emotion', 'negemo': 'negative emotion', 'anx': 'anxiety', 'anger': 'anger', 'sad': 'sadness', 'polarity': 'polarity', 'subjectivity': 'subjectivity', '!': '!', '?': '?', '...': '...', ',': ',', ';': ';', ':': ':', 'total': 'punctuations', 'nb_char': 'number of characters', 'nb_word': 'number of words', 'nb_sent': 'number of sentences', 'nb_para': 'number of paragraphs', 'avg_char_word': 'average number of characters per word', 'avg_word_sent': 'average number of words per sentence', 'avg_sent_para': 'average number of sentences per paragraph', 'cogproc': 'cognitive processes', 'insight': 'insight', 'cause': 'causation', 'discrep': 'discrepancy', 'tentat': 'tentative', 'certain': 'certainty', 'differ': 'differentiation', 'percept': 'perceptual processes', 'see': 'seeing vocabulary', 'hear': 'hearing vocabulary', 'feel': 'feeling vocabulary', 'fkgl': 'Flesch-Kincaid grade level', 'fkre': 'Flesch-Kincaid reading ease', 'dc': 'Dale-Chall formula', 'smog': 'SMOG grade', 'cli': 'Coleman-Liau index', 'ari': 'automated readability index', 'fc': 'FORCAST formula'}

def pres(nav):
    st.write(nav)
    
def glob(nav):
    # SIDEBAR
    st.sidebar.title(nav)
    
    analyze = st.sidebar.radio("Features Interpretability", list(ANALYZE.keys())[:-1])
    
    # CONTENT
    st.header(analyze + " analyze")
    shap_summary_plot(ANALYZE[analyze][2], analyze, ANALYZE[analyze][3])
    
    
def example(nav):
    # SIDEBAR
    st.sidebar.title(nav)
    example_type = st.sidebar.radio("", ["Testing set", "New article"])
    analyze = st.sidebar.multiselect("Features Interpretability", list(ANALYZE.keys())[1:])
    
    # CONTENT
    if example_type == "Testing set":
        ex_testing(analyze)
    elif example_type == "New article":
        ex_new(analyze)

def shap_explainer(model, feat_df):
    # for XGBoost model_output="raw" is the log odds ratio
    explainer = shap.TreeExplainer(model, model_output="raw")
    shap_values = explainer.shap_values(feat_df)
    article_data = (model, explainer, shap_values, feat_df)
    
    return article_data
    
def inference_cfg(article_df):
    article_feat_df = helper.cfg_feat(article_df, nlp, "model/nlp_model", None)
    
    feat_name = pipes["cfg"][0]["count"].get_feature_names()
    feat_cfg = list(np.array(feat_name)[pipes["cfg"][1]["select"].get_support()])
    feat_cfg = helper.production_except_remove(feat_cfg)
    
    article_cfg = pipes["cfg"].transform(article_feat_df.text_cfg_sep_str) 
    article_cfg_df = pd.DataFrame(article_cfg.todense(), columns=feat_cfg)
    
    cfg_proba = pd.DataFrame(models["cfg"].predict_proba(article_cfg_df))[1]
    
    cfg_shap = shap_explainer(models["cfg"], article_cfg_df)
    
    return cfg_proba, cfg_shap
    
def inference_lex(article_df):
    article_lex = pipes["lex"].transform(article_df.text)
    
    article_lex_df = pd.DataFrame(article_lex.todense(), columns=pipes["lex"]["count"].get_feature_names())
    
    lex_proba = pd.DataFrame(models["lex"].predict_proba(article_lex_df))[1]
    
    lex_shap = shap_explainer(models["lex"], article_lex_df)
    
    return lex_proba, lex_shap

def inference_cat(article_df):
    article_feat_df = helper.cat_feat(article_df, nlp, empath, "model/nlp_model", None)
    
    article_cat = pipes["cat"].transform(article_feat_df)
    
    cat_feat_name = []
    for pipe_cat in pipes["cat"].transformer_list:
        cat_feat_name += pipe_cat[1]["dict"].get_feature_names()
    
    article_cat_df = pd.DataFrame(article_cat.todense(), columns=cat_feat_name)
    
    cat_proba = pd.DataFrame(models["cat"].predict_proba(article_cat_df))[1]
    
    cat_shap = shap_explainer(models["cat"], article_cat_df)
    
    return cat_proba, cat_shap

def inference_bert(article_df):
    nb_tokens = 512
    head_size = 256
    col = {"text": "content"}
    bert_model_name = "bert-base-uncased"
    bert_tokenizer = hf.BertTokenizer.from_pretrained(bert_model_name)
    
    article_bert_df = helper.extend_data(article_df.rename(columns=col), tokenizer=bert_tokenizer, nb_tokens=nb_tokens,  mask_f=helper.sent_mask_trunc, arg_mask=head_size)
    article_bert_data = helper.encode_plus(article_bert_df, bert_tokenizer, nb_tokens)
    
    bert_proba = pd.DataFrame(models["bert"].predict(article_bert_data[0]))[0]
    
    return bert_proba
    
def ex_new(analyze):
    user_input = st.text_area("Please enter an article to analyze", 
        "The Supreme Court's rejection of Donald Trump's attempts to keep his tax returns out of the hands of Manhattan District Attorney Cyrus Vance is a potent reminder of this fact: The former President is in real legal jeopardy -- on a number of fronts. In the narrowest aperture what the court's ruling means is that Vance -- and the grand jury he has empaneled to look into hush money payments made in the run-up to the 2016 election to two women alleging affairs with Trump -- will get a look at Trump's most carefully guarded secret: His financial documents. Which is a blow for Trump -- especially given a) how hard he fought the release of these documents, which include years of tax returns, to Vance and b) what we already know about Trump's involvement in the hush money scheme.  In federal search warrant documents released in July 2019, Trump is named for his involvement in the hush money payments. As CNN's Kara Scannell and Marshall Cohen wrote at the time: ‘The documents are the first time that the US authorities have identified Trump by name and allege his involvement at key steps in the campaign finance scheme. Authorities had previously referred to Trump in court filings as 'Individual 1,' the person who directed [Michael] Cohen to make the payments. Trump has publicly denied making the payments. Cohen pleaded guilty to two campaign finance crimes, among others, and is serving a three year prison sentence.’ Cohen, who served as Trump's personal lawyer and fixer for years before turning on him, testified under oath in front of Congress in February 2019 that Trump had personally instructed him to pay Stormy Daniels and Karen McDougal for their silence and that there was ‘no doubt in his mind’ that the candidate knew what he was doing. We know very few details -- beyond that he is looking into the hush money payments -- about Vance's investigation. ‘The work continues,’ Vance tweeted cryptically after the Supreme Court announces its decision Monday on Trump's tax returns. But what we do know is that the Manhattan District Attorney investigation is far from the only legal matter in which Trump currently finds himself entangled. Consider: 1. The New York attorney general's office is looking into how the Trump organization valued its assets. 2. Defamation lawsuits from E. Jean Carroll and Summer Zervos. 3. A fraud lawsuit filed by Trump niece Mary Trump. 4. A possible charge of incitement by the DC attorney general for Trump's role in the January 6 riot at the US Capitol. 5. Two investigations into Trump's attempts to pressure Georgia elected officials to overturn the state's election results. And what we also know is that the freedom from prosecution afforded to a sitting president doesn't hold for a former president. Nor is dealing with all of these various lawsuits cheap. Nor does Trump have unlimited resources at his disposal. (In fact, his financial concerns may be even more pressing than his legal ones.) Trump, of course, is also not unfamiliar with drawn-out court fights. He's often bragged about his ability to use the legal system to his advantage, in fact. ‘Does anyone know more about litigation than Trump?’ Trump said of himself on the campaign trail in 2016. ‘I'm like a Ph.D. in litigation.’  The question going forward, however, is whether Trump's old legal tactics -- delay, delay, delay and hope the other side loses interest or runs out of money -- will work against the considerable number of foes aligned against him. After all, Trump's no longer just some rich guy. He's the former president of the United States. Which puts a MUCH larger legal target on his back.", 
        height=None)
    
    article_df = pd.DataFrame({"text_to_clean":[user_input]})
    article_df["text"] = helper.clean_fake(article_df["text_to_clean"].apply(lambda t: helper.clean_content(helper.clean_base(t))))
    
    if st.sidebar.button('Analyze'):
    
        cfg_proba, cfg_shap = inference_cfg(article_df)
        lex_proba, lex_shap = inference_lex(article_df)
        cat_proba, cat_shap = inference_cat(article_df)
        bert_proba = inference_bert(article_df)
        
        article_feat_proba = pd.DataFrame(
            {"lex": lex_proba, 
             "cfg": cfg_proba,
             "cat": cat_proba,
             "bert": bert_proba,
            })
        
        ml_shap = shap_explainer(models["meta_learner"], article_feat_proba)
        top_ml_shap = meta_df(0, ml_shap, True).rename(index={'cat':'sem', 'bert':'Bert'})
        proba_ml, predict_ml = predict(0, models["meta_learner"], article_feat_proba)
        
        st.markdown("***")
        plot_class(top_ml_shap, proba_ml)
        
        if analyze:
            st.markdown("***")
            plot_all_proba(0, top_ml_shap, analyze)
        
        article_text = article_df.text.values[0]

        st.markdown("***")
        if "Lexical" in analyze:
            top_shap = meta_df(0, lex_shap)
            article_text = lexical_sent(article_text, top_shap)
                
        with st.beta_expander("Article"):
            st.write(f"{article_text}", unsafe_allow_html=True)

        if "Categorized" in analyze:
            top_shap = meta_df(0, cat_shap)
            with st.beta_expander("Categorized"):
                    sem_ana_empath(top_shap)

        if "Grammatical" in analyze:
            top_shap = meta_df(0, cfg_shap)
            with st.beta_expander("Grammatical"):
                syn_ana(top_shap) 
              
    
def ex_testing(analyze):
    testing_article = st.selectbox("Please select an article to analyze", 
        demo_df.title.values)
    
    article = demo_df[demo_df.title == testing_article]
    article_id = article.index[0]
    
    top_ml_shap = meta_df(article_id, ANALYZE["Classification"], True)
    proba_ml, predict_ml = predict(article_id, ANALYZE["Classification"][0], ANALYZE["Classification"][3])
    
    if demo_df.label_int.iloc[article_id] == 0:
        badge_label = "[![Generic badge](https://img.shields.io/badge/label-REAL-success.svg)](https://shields.io/)"
    else:
        badge_label = "[![Generic badge](https://img.shields.io/badge/label-FAKE-red.svg)](https://shields.io/)"
        
    if predict_ml == 0:
        badge_pred = "[![Generic badge](https://img.shields.io/badge/prediction-REAL-success.svg)](https://shields.io/)"
    else:
        badge_pred = "[![Generic badge](https://img.shields.io/badge/prediction-FAKE-red.svg)](https://shields.io/)"
    
    st.markdown(f"### {testing_article}"
               f" ({article_id})")
        
    st.markdown(f"{badge_label} / {badge_pred}")
    #st.write(article_id)
    st.markdown("***")
    
    plot_class(top_ml_shap, proba_ml)
    
    if analyze:
        st.markdown("***")
        plot_all_proba(article_id, top_ml_shap, analyze)
    
    article_text = article.text.values[0]
    
    st.markdown("***")
    if "Lexical" in analyze:
        top_shap = meta_df(article_id, ANALYZE["Lexical"])
        #type_article = st.radio("Please select the lexical visualization:", ("Word", "Sentence mean", "Sentence norm"), index=2)
        #article_word = lexical_text(article_text, top_shap)
        #article_sent_mean = lexical_sent_mean(article_text, top_shap)
        article_text = lexical_sent(article_text, top_shap)
            
    with st.beta_expander("Article"):
        st.write(f"{article_text}", unsafe_allow_html=True)

            
    if "Categorized" in analyze:
        top_shap = meta_df(article_id, ANALYZE["Categorized"])
        with st.beta_expander("Categorized"):
                sem_ana(top_shap)
        #st.markdown("***")

    if "Grammatical" in analyze:
        top_shap = meta_df(article_id, ANALYZE["Grammatical"])
        with st.beta_expander("Grammatical"):
            syn_ana(top_shap)
        #st.markdown("***")   
 
 
GOTO = {
    #"Presentation": pres, 
    "Global Analyze": glob, 
    "Example": example
}

def plot_all_proba(article_id, top_ml_shap, analyze):
    fid = top_ml_shap.sort_values(["abs_shap"], ascending=False).index
    features = [f for f in fid if names[f] in analyze]
    
    fig_probas = plt.figure(figsize=(8, 3), constrained_layout=True)
    gs = gridspec.GridSpec(2, 4)
    
    if len(analyze) == 1:
        sp = [gs[1:, :]]
    elif len(analyze) == 2:
        sp = [gs[:, :2], gs[:, 2:]]
    elif len(analyze) == 3:
        sp = [gs[:1, 1:3], gs[1:, :2], gs[1:, 2:]]
    elif len(analyze) == 4:
        sp = [gs[:1, :2], gs[:1, 2:], gs[1:, :2], gs[1:, 2:]]
        
    for f, sid in zip(features, sp):
        fake_proba = top_ml_shap.loc[f].fake_proba
        proba = [1-fake_proba, fake_proba]
        ax = fig_probas.add_subplot(sid, aspect=20)
        plot_proba2(proba, ax, names[f])
    
    #plt.show()
    plt.tight_layout()
    st.pyplot(fig_probas)
    plt.clf()       

def predict(j, model, feat_df):
    proba = model.predict_proba(feat_df.iloc[[j]])[0]
    pred = model.predict(feat_df.iloc[[j]])[0]
    return proba, pred

def donut_color(shap, shap_max):
    norm_color = colors.Normalize(vmin=-0.05, vmax=shap_max*1.25)
    if shap < 0:
        color = colors.to_hex(cm.Greens(norm_color(np.abs(shap))))
    else:
        color = colors.to_hex(cm.Reds(norm_color(shap)))
    return color

def donut_labeling(label, shap):
    text_label = "{}: {:.2f}%"
    return text_label.format(label, shap)
    
def meta_df(j, meta, color=False):
    _, _, shap_values_val, val_features = meta
    meta_df = pd.DataFrame({"fake_proba": val_features.iloc[j]})
    meta_df["shap"] = shap_values_val[j]
    meta_df["abs_shap"] = meta_df.shap.apply(abs)
    meta_df["perc_shap"] = meta_df.abs_shap.apply(lambda s: s * 100 / meta_df.abs_shap.sum())
    if color:
        meta_df["shap_color"] = meta_df.shap.apply(lambda s: donut_color(s, meta_df.abs_shap.max()))
    return meta_df
    
def plot_class(top_ml_shap, proba):
    fig = plt.figure(figsize=(8, 2))
    
    ax1 = fig.add_subplot(1, 2, 1)
    plot_donut2(top_ml_shap, ax1)
    ax2 = fig.add_subplot(1, 2, 2, aspect=20)
    plot_proba2(proba, ax2, "Total")
    
    plt.show()
    st.pyplot(fig, bbox_inches="tight")
    plt.clf()
    
def plot_donut2(top_ml_shap, ax):
    names = {"lex": "Lexical", "cfg": "Grammatical", "sem": "Categorized", "Bert": "Bert"}
    labels = [donut_labeling(names[s[0]], s[1].perc_shap) for s in top_ml_shap.iterrows()]
    
    size = 0.4

    ax.pie(top_ml_shap.perc_shap, labels=labels, radius=1, colors=top_ml_shap.shap_color, wedgeprops=dict(width=size, edgecolor='w', linewidth=3), textprops=dict(size="medium"))
    
    ax.set_title("Features Influence", fontsize="large")

def plot_proba2(proba, ax, title):
 
    real_proba = round(proba[0]*100,2)
    fake_proba = round(proba[1]*100,2)
    plt.barh([0], [real_proba - 0.5], color='tab:green', height=0.5, label="Real")
    plt.barh([0], [fake_proba - 0.5], left=[real_proba + 0.5], color='tab:red', height=0.5, label="Fake")
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(-1, 101)
    ax.set_ylim(-0.31, 0.31)
    
    text_real = ax.text(real_proba/2, 0, str(real_proba) + "%", fontsize="medium", ha="center", va="center", color="white")
    text_fake = ax.text(real_proba + fake_proba/2, 0, str(fake_proba) + "%", fontsize="medium", ha="center", va="center", color="white")
    
    if real_proba < 10:
        text_real.set_visible(False)
    elif fake_proba < 10:
        text_fake.set_visible(False)
        
    ax.legend(ncol=2, bbox_to_anchor=(0.5, 0),
                  loc='upper center', fontsize='medium')
    ax.set_title(title + " Reliability Score", fontsize="large", fontweight="medium", va="baseline")

def show_cfg_tags():
    show_tags = st.checkbox("Show grammatical tags")
    if show_tags:
        st.image(cfg_tags, caption="Grammatical tags")
        
def shap_summary_plot(shap_values, analyze, feat_df):
    title = "SHAP Summary plot for " + analyze + " Important Features:"
    st.subheader(title)
    # change lex, cfg, sem for global classification summary plot
    feat_df = feat_df.rename(columns={**names, **sem_dict})
    ### TO CHANGE
    if analyze == "Lexical":
        scid = feat_df.columns.get_loc("story continued")
        shap_values[:, scid] = 0
    fig, ax = plt.subplots()
    ax = shap.summary_plot(shap_values, feat_df, max_display=10, plot_type="bar", show=False)
    #shap.summary_plot(shap_values, feat_df, max_display=10, show=False)
    st.pyplot(fig, bbox_inches='tight')
    plt.clf()
    if analyze == "Grammatical":
        #show_cfg_tags()
        with st.beta_expander("Grammatical Tags"):
            st.image(cfg_tags)

def shap_plot(j, explainer, shap_values, feat_df):
    shap.force_plot(explainer.expected_value, shap_values[j],
            feat_df.iloc[[j]], link="logit", matplotlib=True, show=False)
    st.pyplot(bbox_inches='tight')
    plt.clf()

def lexical_sent(article, top):
    #top = top[(top.abs_shap > 0) & (top.fake_proba > 0)]
    top = top[top.fake_proba > 0]
    sentences = sent_tokenize(re.sub(r'\s+', ' ', article))
    sent_shap_values = [{"sent":sent, "shap_values":[]} for sent in sentences]
    shap_mean = 0
    
    for top_r in top.iterrows():
        text = top_r[0]
        shap = top_r[1].iloc[1]
        score = round(top_r[1].iloc[0],2)
        
        word_to_print = "Lexical Score: {} &#13;SHAP Score: {}".format(score, round(shap, 6))
        span = r'<u title="{}">\g<0></u>'.format(word_to_print)
        src_str  = re.compile(r"\b{}\b".format(text), re.IGNORECASE)
        
        for ds in sent_shap_values:
            all_matches = src_str.findall(ds["sent"])
            for match in all_matches:
                ds["shap_values"].append(shap)
                shap_mean += shap 
            
            if np.abs(shap) > 0:
                ds["sent"] = src_str.sub(span, ds["sent"]) 
    
    nb_words_shap = sum([len(ds["shap_values"]) for ds in sent_shap_values])
    #nb_words = len(article.split())
    shap_mean /= nb_words_shap
    shap_std_tot = 0
    
    for ds in sent_shap_values:
        shap_std_sent = 0
        shap_mean_sent = 0
        for shap in ds["shap_values"]:
            shap_std_sent += (shap - shap_mean)**2
            shap_mean_sent += shap
        
        if len(ds["shap_values"]) == 0:
            ds["shap_std_sent"] = 0
            ds["shap_mean_sent"] = 0
        else:
            ds["shap_std_sent"] = np.sqrt(shap_std_sent / len(ds["shap_values"])) 
            ds["shap_mean_sent"] = shap_mean_sent / len(ds["shap_values"])
            
        shap_std_tot += shap_std_sent
        
    shap_std_tot = np.sqrt(shap_std_tot / nb_words_shap)
    
    shap_sent = []
    for ds in sent_shap_values:
        sign = np.sign(ds["shap_mean_sent"])
        if sign == 0: sign = 1
        ds["shap_sent"] = (sign * ds["shap_std_sent"]) / shap_std_tot
        shap_sent.append(ds["shap_sent"])

    norm_text = colors.Normalize(vmin=0, vmax=np.abs(np.array(shap_sent).max()))
   
    new_article = ""
    
    for ds in sent_shap_values:
        to_print = "SHAP norm: {}".format(round(ds["shap_sent"], 6))
                                                               
        if ds["shap_sent"] < 0:
            color = colors.to_hex(cm.Greens(norm_text(np.abs(ds["shap_sent"]))))
        else:
            color = colors.to_hex(cm.Reds(norm_text(np.abs(ds["shap_sent"]))))
        
        cmap_text = colors.ListedColormap(["black", "black", "black", "white"])
        color_text = colors.to_hex(cmap_text(norm_text(np.abs(ds["shap_sent"]))))
        
        span = r'<span title="{}" style="background-color:{}; color:{}">{}</span>'
        
        span_c = span.format(to_print, color, color_text, ds["sent"])
        new_article += (" " + span_c)
     
    return new_article
    
def lexical_sent_mean(article, top):
    #top = top[(top.abs_shap > 0) & (top.fake_proba > 0)]
    top = top[top.fake_proba > 0]
    sentences = sent_tokenize(re.sub(r'\s+', ' ', article))
    sent_shap_values = [{"sent":sent, "shap_values":[]} for sent in sentences]
    shap_mean = 0
    
    for top_r in top.iterrows():
        text = top_r[0]
        shap = top_r[1].iloc[1]
        score = round(top_r[1].iloc[0],2)
        
        word_to_print = "Lexical Score: {} &#13;SHAP Score: {}".format(score, round(shap, 6))
        span = r'<u title="{}">\g<0></u>'.format(word_to_print)
        src_str  = re.compile(r"\b{}\b".format(text), re.IGNORECASE)
        
        for ds in sent_shap_values:
            all_matches = src_str.findall(ds["sent"])
            for match in all_matches:
                ds["shap_values"].append(shap)
                shap_mean += shap 
            
            if np.abs(shap) > 0:
                ds["sent"] = src_str.sub(span, ds["sent"])

            
    nb_words_shap = sum([len(ds["shap_values"]) for ds in sent_shap_values])
    #nb_words = len(article.split())
    shap_mean /= nb_words_shap
    
    for ds in sent_shap_values:
        shap_mean_sent = 0
        for shap in ds["shap_values"]:
            shap_mean_sent += shap
        ds["shap_mean_sent"] = shap_mean_sent / len(ds["shap_values"])

    shap_mean_sent = np.array([ds["shap_mean_sent"] for ds in sent_shap_values])
    norm_text = colors.Normalize(vmin=0, vmax=np.abs(shap_mean_sent.max()))
   
    new_article = ""
    
    for ds in sent_shap_values:
        to_print = "SHAP mean: {}".format(round(ds["shap_mean_sent"], 6))
                                                               
        if ds["shap_mean_sent"] < 0:
            color = colors.to_hex(cm.Greens(norm_text(np.abs(ds["shap_mean_sent"]))))
        else:
            color = colors.to_hex(cm.Reds(norm_text(np.abs(ds["shap_mean_sent"]))))
        
        cmap_text = colors.ListedColormap(["black", "black", "black", "white"])
        color_text = colors.to_hex(cmap_text(norm_text(np.abs(ds["shap_mean_sent"]))))
        
        span = r'<span title="{}" style="background-color:{}; color:{}">{}</span>'
        
        span_c = span.format(to_print, color, color_text, ds["sent"])
        new_article += (" " + span_c)
            
    return new_article
 

    
def lexical_text(article, top):
    article = re.sub(r'\s+', ' ', article)
    #top = top[top.abs_shap > 0.05]
    top = top[(top.abs_shap > 0) & (top.fake_proba > 0)]
    norm_real = colors.Normalize(vmin=0, vmax=np.abs(top.iloc[:, 1].min()))
    norm_fake = colors.Normalize(vmin=0, vmax=top.iloc[:, 1].max())
    norm_text = colors.Normalize(vmin=0, vmax=np.abs(top.iloc[:, 1]).max())
    
    for top_r in top.iterrows():
        text = top_r[0]
        score = round(top_r[1].iloc[0],2)
        shap = top_r[1].iloc[1]
        to_print = "{} &#13;Lexical Score: {} &#13;SHAP Score: {}".format(text, score, round(shap, 6))
                                                               
        if shap < 0:
            color = colors.to_hex(cm.Greens(norm_text(np.abs(shap))))
        else:
            color = colors.to_hex(cm.Reds(norm_text(np.abs(shap))))
        
        color_text = colors.to_hex(cm.gray(norm_text(np.abs(shap))))
        
        span = r'<span title="{}" style="background-color:{}; color:{}">\g<0></span>'
        
        src_str  = re.compile(r"\b{}\b".format(text), re.IGNORECASE)
        
        span_c = span.format(to_print, color, color_text)
        #st.write(span_c)
        article = src_str.sub(span_c, article)
        #st.write(to_change)
        #for tc in set(to_change):
        #    st.write(tc)
        #    article = article.replace(tc, span.format(to_print, color, color_text, tc))
    return article

def sem_donut():
    fig, ax = plt.subplots()

    size = 0.3
    vals = np.array([[60., 32.], [37., 40.], [29., 10.]])

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3)*4)
    inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

    ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
           wedgeprops=dict(width=size, edgecolor='w'))

    ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
           wedgeprops=dict(width=size, edgecolor='w'))

    ax.set(aspect="equal", title='Pie plot with `ax.pie`')
    plt.show()

def sem_ana_empath(shap_df):
    shap_df_cp = shap_df.copy()
    shap_df_cp["sem_cat"] = "init"
    shap_df_cp["sem_glob_cat"] = "init"
    
    ## QUALITY = Language Level
    # Informality
    informal = ["swearing_terms", "swear", "filler", "netspeak", "assent", "nonflu"]
    # Diversity
    div_words = ["diff_words", "uni_words"]
    div_pos = ["diff_noun_n", "diff_noun_w", "uni_noun_n", "uni_noun_w", 
           "diff_verb_n", "diff_verb_w", "uni_verb_n", "uni_verb_w", 
           "diff_adj_n", "diff_adj_w", "uni_adj_n", "uni_adj_w", 
           "diff_adv_n", "diff_adv_w", "uni_adv_n", "uni_adv_w"]
    # Subjectivity
    subj = ["bias", "factive", "report"]
    
    # Readability
    read = ["fkgl", "fkre", "dc", "smog", "cli", "ari", "fc"]
    
    quality_dict = {"Informality": shap_df.loc[informal].perc_shap.sum(), 
               "Diversity": shap_df.loc[div_words + div_pos].perc_shap.sum(), 
               "Subjectivity": shap_df.loc[subj].perc_shap.sum(),
               "Readability": shap_df.loc[read].perc_shap.sum()}
    
    shap_df_cp["sem_cat"].loc[informal] = "Informality"
    shap_df_cp["sem_glob_cat"].loc[informal] = "Language Level"
    
    shap_df_cp["sem_cat"].loc[div_words + div_pos] = "Diversity"
    shap_df_cp["sem_glob_cat"].loc[div_words + div_pos] = "Language Level"
    
    shap_df_cp["sem_cat"].loc[subj] = "Subjectivity"
    shap_df_cp["sem_glob_cat"].loc[subj] = "Language Level"
    
    shap_df_cp["sem_cat"].loc[read] = "Readability"
    shap_df_cp["sem_glob_cat"].loc[read] = "Language Level"
    
    ## SENSATIONALISM
    # Sentiment
    sentiment_pos_cat = ["positive_emotion", "cheerfulness", "love", "trust", "affection", "joy"]
    sentiment_pos_bonus_cat = ["attractive", "optimism", "healing", "sympathy", "valuable", "fun", "contentment", "beauty"]

    sentiment_neg_cat = ["negative_emotion", "hate", "fear", "suffering", "nervousness", "irritability", "disgust", "sadness", "anger", "disappointment", "exasperation", "deception"]
    sentiment_neg_bonus_cat = ["pain", "torment", "envy", "rage", "injury", "crime", "dispute", "weakness", "kill", "aggression", "death", "violence", "neglect", "fight", "shame"]
    
    affect = sentiment_pos_cat + sentiment_pos_bonus_cat + sentiment_neg_cat + sentiment_neg_bonus_cat
    sent_nltk = ["polarity", "subjectivity"]
    # Punctuation
    punct = ["!", "?", "...", ",", ";", ":", "total"]
    
    sensat_dict = {"Sentiment": shap_df.loc[affect + sent_nltk].perc_shap.sum(), 
              "Punctuation": shap_df.loc[punct].perc_shap.sum()}

    shap_df_cp["sem_cat"].loc[affect + sent_nltk] = "Sentiment"
    shap_df_cp["sem_glob_cat"].loc[affect + sent_nltk] = "Sensationalism"
    
    shap_df_cp["sem_cat"].loc[punct] = "Punctuation"
    shap_df_cp["sem_glob_cat"].loc[punct] = "Sensationalism"
    
    ## QUANTITY = Text Structure
    quant = ["nb_char", "nb_word", "nb_sent", "nb_para", "avg_char_word", "avg_word_sent", "avg_sent_para"]
    
    shap_df_cp["sem_cat"].loc[quant] = "Text Structure"
    shap_df_cp["sem_glob_cat"].loc[quant] = "Text Structure"
    
    # COGNITIVE DIMENSION
    cogproc_cat = ["cogproc", "insight", "cause", "discrep", "tentat", "certain"]
    cogproc_bonus_cat = ["anticipation", "negotiate"]
    
    cogn_process = cogproc_cat + cogproc_bonus_cat
    
    shap_df_cp["sem_cat"].loc[cogn_process] = "Cognitive Dimension"
    shap_df_cp["sem_glob_cat"].loc[cogn_process] = "Cognitive Dimension"
    # PERCEPTUAL DIMENSION
    percept_cat = ["percept", "see", "hear", "feel", "hearing", "listen", "emotional"]
    percept_bonus_cat = ["smell", "warmth", "lust", "surprise", "confusion"]
    
    perc_process = percept_cat + percept_bonus_cat
         
    shap_df_cp["sem_cat"].loc[perc_process] = "Perceptual Dimension"
    shap_df_cp["sem_glob_cat"].loc[perc_process] = "Perceptual Dimension"
        
    # Sum shap values by cat
    shap_glob_df = shap_df_cp[["sem_glob_cat", "shap"]].groupby(["sem_glob_cat"]).sum().reset_index()
    shap_glob_df["abs_shap"] = np.abs(shap_glob_df.shap)
    shap_glob_df["color"] = shap_glob_df.shap.apply(lambda s: donut_color(s, shap_glob_df.abs_shap.max()))
    shap_glob_df = shap_glob_df.sort_values(["abs_shap"], ascending=False).reset_index()
      
    ###############################
    fig = plt.figure(figsize=(6, 3.5))
    
    ax = sns.barplot(x="shap", y="sem_glob_cat", data=shap_glob_df, orient="h", palette=shap_glob_df.color, dodge=False)
    ax.set(xlabel="Categorized Features influence (Real/Fake)", ylabel="Categorized Features")
    fig.suptitle("Categorized Features Influence", fontsize="large")
    plt.show()
    st.pyplot(fig, bbox_inches='tight')
    plt.clf()
    ###############################
    
def sem_ana(shap_df):
    shap_df_cp = shap_df.copy()
    shap_df_cp["sem_cat"] = "init"
    shap_df_cp["sem_glob_cat"] = "init"
    
    ## QUALITY = Language Level
    # Informality
    informal = ["informal", "swear", "netspeak", "assent", "nonflu", "filler"]
    # Diversity
    div_words = ["diff_words", "uni_words"]
    div_pos = ["diff_noun_n", "diff_noun_w", "uni_noun_n", "uni_noun_w", 
           "diff_verb_n", "diff_verb_w", "uni_verb_n", "uni_verb_w", 
           "diff_adj_n", "diff_adj_w", "uni_adj_n", "uni_adj_w", 
           "diff_adv_n", "diff_adv_w", "uni_adv_n", "uni_adv_w"]
    # Subjectivity
    subj = ["bias", "factive", "report"]
    
    # Readability
    read = ["fkgl", "fkre", "dc", "smog", "cli", "ari", "fc"]
    
    quality_dict = {"Informality": shap_df.loc[informal].perc_shap.sum(), 
               "Diversity": shap_df.loc[div_words + div_pos].perc_shap.sum(), 
               "Subjectivity": shap_df.loc[subj].perc_shap.sum(),
               "Readability": shap_df.loc[read].perc_shap.sum()}
    
    shap_df_cp["sem_cat"].loc[informal] = "Informality"
    shap_df_cp["sem_glob_cat"].loc[informal] = "Language Level"
    
    shap_df_cp["sem_cat"].loc[div_words + div_pos] = "Diversity"
    shap_df_cp["sem_glob_cat"].loc[div_words + div_pos] = "Language Level"
    
    shap_df_cp["sem_cat"].loc[subj] = "Subjectivity"
    shap_df_cp["sem_glob_cat"].loc[subj] = "Language Level"
    
    shap_df_cp["sem_cat"].loc[read] = "Readability"
    shap_df_cp["sem_glob_cat"].loc[read] = "Language Level"
    
    ## SENSATIONALISM
    # Sentiment
    affect = ["affect", "posemo", "negemo", "anx", "anger", "sad"]
    sent_nltk = ["polarity", "subjectivity"]
    # Punctuation
    punct = ["!", "?", "...", ",", ";", ":", "total"]
    
    sensat_dict = {"Sentiment": shap_df.loc[affect + sent_nltk].perc_shap.sum(), 
              "Punctuation": shap_df.loc[punct].perc_shap.sum()}

    shap_df_cp["sem_cat"].loc[affect + sent_nltk] = "Sentiment"
    shap_df_cp["sem_glob_cat"].loc[affect + sent_nltk] = "Sensationalism"
    
    shap_df_cp["sem_cat"].loc[punct] = "Punctuation"
    shap_df_cp["sem_glob_cat"].loc[punct] = "Sensationalism"
    
    ## QUANTITY = Text Structure
    quant = ["nb_char", "nb_word", "nb_sent", "nb_para", "avg_char_word", "avg_word_sent", "avg_sent_para"]
    
    shap_df_cp["sem_cat"].loc[quant] = "Text Structure"
    shap_df_cp["sem_glob_cat"].loc[quant] = "Text Structure"
    
    # COGNITIVE DIMENSION
    cogn_process = ["cogproc", "insight", "cause", "discrep", "tentat", "certain", "differ"]
    
    shap_df_cp["sem_cat"].loc[cogn_process] = "Cognitive Dimension"
    shap_df_cp["sem_glob_cat"].loc[cogn_process] = "Cognitive Dimension"
    # PERCEPTUAL DIMENSION
    perc_process = ["percept", "see", "hear", "feel"]
         
    shap_df_cp["sem_cat"].loc[perc_process] = "Perceptual Dimension"
    shap_df_cp["sem_glob_cat"].loc[perc_process] = "Perceptual Dimension"
        
    # Sum shap values by cat
    shap_glob_df = shap_df_cp[["sem_glob_cat", "shap"]].groupby(["sem_glob_cat"]).sum().reset_index()
    shap_glob_df["abs_shap"] = np.abs(shap_glob_df.shap)
    shap_glob_df["color"] = shap_glob_df.shap.apply(lambda s: donut_color(s, shap_glob_df.abs_shap.max()))
    shap_glob_df = shap_glob_df.sort_values(["abs_shap"], ascending=False).reset_index()
      
    ###############################
    fig = plt.figure(figsize=(6, 3.5))
    
    ax = sns.barplot(x="shap", y="sem_glob_cat", data=shap_glob_df, orient="h", palette=shap_glob_df.color, dodge=False)
    ax.set(xlabel="Categorized Features influence (Real/Fake)", ylabel="Categorized Features")
    fig.suptitle("Categorized Features Influence", fontsize="large")
    plt.show()
    st.pyplot(fig, bbox_inches='tight')
    plt.clf()
    ###############################
    

def sem_color(row):
    norm_color = colors.Normalize(vmin=-20, vmax=60)
    if row.shap < 0:
        color = colors.to_hex(cm.Greens(norm_color(row.perc_shap)))
    else:
        color = colors.to_hex(cm.Reds(norm_color(row.perc_shap)))
    return color

def syn_ana(shap_df):
    
    fig = plt.figure(figsize=(10, 8))
    
    shap_df_cp = shap_df.copy()
    shap_df_cp = shap_df_cp.sort_values(["abs_shap"], ascending=False).reset_index()[:20]
    shap_df_cp["color"] = shap_df_cp.shap.apply(lambda s: donut_color(s, shap_df_cp.abs_shap.max()))

    ax = sns.barplot(x="shap", y="index", data=shap_df_cp, orient="h", palette=shap_df_cp.color, dodge=False)
    ax.set(xlabel="Grammatical Features influence (Real/Fake)", ylabel="Grammatical Features")
    fig.suptitle("Grammatical Features Influence", fontsize="xx-large")
    plt.show()
    st.pyplot(fig, bbox_inches='tight')
    plt.clf()
    
    #show_cfg_tags()
    #with st.beta_expander("Grammatical Tags"):
    st.image(cfg_tags)
    
def main():
    #st.title("Explainable Fake News Detection")
    st.sidebar.title("Explainable Fake News Detection")
    nav = st.sidebar.radio("Go to", list(GOTO.keys()))
    GOTO[nav](nav)
    
if __name__ == "__main__":
    main()
