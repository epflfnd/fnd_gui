import requests

import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image
from io import BytesIO
from matplotlib import cm, colors
from bs4 import BeautifulSoup as bs
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# requests page webpage from the given url
def get_page(url):
    session = requests.Session()
    
    #retry = Retry(connect=5, backoff_factor=0.5)
    #adapter = HTTPAdapter(max_retries=retry)
    #session.mount("http://", adapter)
    #session.mount("https://", adapter)
    
    try:
        return session.get(url, timeout=2)
    except Exception: 
        return None

# Get content of webpage with removing header and footer content
def get_body(soup):
    body_p = soup.find("body").find_all("p")
    header_p = soup.find("header").find_all("p")
    footer_p = soup.find("footer").find_all("p")
    
    if body_p is not None:
        return " ".join([p.text for p in body_p if (p not in header_p) and (p not in footer_p)])
    else:
        return None
    
# get title of article
def get_title(soup):
    body_h = soup.find("body").find_all(["h1", "h2", "h3"])
    header_h = soup.find("header").find_all(["h1", "h2", "h3"])
    footer_h = soup.find("footer").find_all(["h1", "h2", "h3"])
    
    if body_h is not None:
        for h in body_h:
            title_cond = ("title" in str(h)) or ("heading" in str(h))
            if (h not in header_h) and (h not in footer_h) and title_cond:
                return h.text.strip()
    
    return ""

# plot proba bar for the given proba
def plot_proba(proba, ax, title):
    
    real_proba = round(proba,2)
    fake_proba = round(100 - proba,2)
    
    plt.barh([0], [real_proba - 0.5], color='tab:green', height=0.5, label="Real")
    plt.barh([0], [fake_proba - 0.5], left=[real_proba + 0.5], color='tab:red', height=0.5, label="Fake")
    
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(-1, 101)
    ax.set_ylim(-0.31, 0.31)
    
    text_real = ax.text(real_proba/2, 0, str(real_proba) + "%", fontsize="medium", ha="center", va="center", color="white")
    text_fake = ax.text(real_proba + fake_proba/2, 0, str(fake_proba) + "%", fontsize="medium", ha="center", va="center", color="white")
    
    if real_proba < 15:
        text_real.set_visible(False)
    elif fake_proba < 15:
        text_fake.set_visible(False)
        
    ax.legend(ncol=2, bbox_to_anchor=(0.5, 0),
                  loc='upper center', fontsize='medium')
    ax.set_title(title + " Reliability Score", fontsize="large", fontweight="medium", va="baseline")

# plot proba bars for each learner
def plot_all_proba(response):
    results = {"Total":response['reliability score'],
          "Lexical":response['lexical results']['reliability score'],
          "Grammatical ":response['grammatical results']['reliability score'],
          "Linguistic Categories":response['categorized results']['reliability score'],
          "Semantic":response['bert results']['reliability score']}
    
    #(8,5) (4,4), [0 2 2 3 3...]
    fig_probas = plt.figure(figsize=(8, 4), constrained_layout=True)
    gs = gridspec.GridSpec(3, 4)
    
    sp = [gs[0, 1:3], gs[1, :2], gs[1, 2:], gs[2, :2], gs[2, 2:]]
    
    for f,sid in zip(results.items(), sp):
        ax = fig_probas.add_subplot(sid, aspect=20)
        plot_proba(f[1], ax, f[0])
    
    #plt.show()
    #plt.tight_layout()
    buf = BytesIO()
    fig_probas.savefig(buf, format="png")
    col1, col2, col3 = st.columns([0.2, 5, 0.2])
    col2.image(buf, use_column_width='auto')

    #st.pyplot(fig_probas)
    #plt.clf()   

# rewrite article with color sentences 
def lex_plot(response):
    sents = list(response.keys())
    shaps = [float(response[k]) for k in sents]
    color_shaps = color_palette(shaps)

    norm_text = colors.Normalize(vmin=0, vmax=np.abs(np.array(shaps).max()))
   
    new_article = ""
    
    for sent, shap, color in zip(sents, shaps, color_shaps):
        to_print = "influence value: {}".format(shap)

        cmap_text = colors.ListedColormap(["black", "black", "black", "white"])
        color_text = colors.to_hex(cmap_text(norm_text(np.abs(shap))))
        
        span = r'<span title="{}" style="background-color:{}; color:{}">{}</span>'
        
        span_c = span.format(to_print, color, color_text, sent)
        new_article += (" " + span_c)
     
    return new_article

# create color palette from shaps shap values
def color_palette(shaps):
    norm_color = colors.Normalize(vmin=-0.05, vmax=max(np.abs(shaps))*1.25)
    color_shaps = []
    for shap in shaps:
        if shap < 0:
            color_shaps.append(colors.to_hex(cm.Reds(norm_color(np.abs(shap)))))
        else:
            color_shaps.append(colors.to_hex(cm.Greens(norm_color(shap))))
            
    return color_shaps

# plot linguistic cat feature influences    
def cat_plot(response):
    cats = list(response.keys())
    shaps = [float(response[k]) for k in cats]
    color_shaps = color_palette(shaps)
    
    fig = plt.figure(figsize=(6, 3.5), constrained_layout=True)
    
    ax = sns.barplot(x=shaps, y=cats, palette=color_shaps, dodge=False)
    
    ax.set(xlabel="Linguistic Categories Features Influence (Real/Fake)", ylabel="Linguistic Categories Features")
    fig.suptitle("Linguistic Categories Features Influence", fontsize="large")
    #plt.show()
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    col1, col2, col3 = st.columns([0.7, 5, 0.2])
    col2.image(buf, use_column_width='auto')
    
    #st.pyplot(fig, bbox_inches='tight')
    #plt.clf()

# increase size of width of page
def _max_width_(prcnt_width:int=75):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )
    
def main():
    _max_width_(prcnt_width=60)
    #st.set_page_config(layout="centered")
    #st.sidebar.title("EBU Fake News detection API")
    col1, col2 = st.columns([1,2])
    with col1:
        ebu_logo = Image.open("ebu_logo_off.png")
        st.image(ebu_logo)
    with col2:
        #st.title("Fake News Detection API")
        st.markdown("<h1 style='text-align: center;'>LynX - News Analyzer</h1>", unsafe_allow_html=True)
    
    with st.form("my_form"):    
        text_area = st.text_area("Insert the URL or text of an article to analyze",  height=None)
        explain_button = st.checkbox("Base learners explainability")
        
        submit_button = st.form_submit_button('Analyze the article', help='Run Analyze')
        #submit_button = st.button('Analyze the article', help='Run Analyze')
        if submit_button:
            
            page = get_page(text_area)
            if (page is None) or (page.status_code != 200):
                st.info("Not a valid URL")
                content = text_area
            else:
                st.success("valid URL")
                soup = bs(page.content, 'html.parser')
                body = get_body(soup)
                title = get_title(soup)
                content = title + " " + body
                
            ex_bool = "true" if explain_button else "false"
            api_add = "http://15.237.83.195:8504/predict?explain={}".format(ex_bool)
            article_json = {"content": content}
            
            with st.spinner('Wait for the analyze'):
                response = requests.post(api_add, json=article_json)
            
            if response.status_code != 200:
                st.error("Not a valid text")
            else:
                #st.write("Response time : {}".format(response.elapsed.total_seconds()))
                #st.write("Reliability score : {}%".format(response.json()))
                #st.write('***')
                
                plot_all_proba(response.json())
                
                if explain_button:
                    
                    article_text = lex_plot(response.json()['lexical results']['explainability'])
                    with st.expander("Article"):
                        st.write(f"{article_text}", unsafe_allow_html=True)
                    
                    with st.expander("Linguistic Categories"):
                        cat_plot(response.json()['categorized results']['explainability'])

            
            
if __name__ == "__main__":
    main()
    
    
    
    
    
    