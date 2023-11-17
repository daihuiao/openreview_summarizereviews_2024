import openreview
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime
import os
from collections import defaultdict
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imageio import imread
from wordcloud import WordCloud
from functools import partial
# sns.set(style='darkgrid', context='talk', palette='colorblind')

first_time = False
if first_time:
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

year = 2024

lemmatizer = WordNetLemmatizer()

excluded = ['via', 'towards', 'based', 'method', 'use', 'framework', 'task', 'learn', 'based',
            'model', 'network', 'neural', 'improve', 'deep', 'multi']

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(title):
    """lemmatize
    e.g., 'learning' -> 'learn'
    """
    word_list = nltk.word_tokenize(title)
    return [lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w)) for w in word_list]


def remove_stopword(title):
    word_split = title
    valid_word = []
    for word in word_split:
        word = word.strip().strip(string.digits)
        if word != "":
            valid_word.append(word)
    word_split = valid_word
    stop_words = set(stopwords.words('english'))
    # add punctuations
    punctuations = list(string.punctuation)
    [stop_words.add(punc) for punc in punctuations]
    # remove null
    stop_words.add("null")
    stop_words.update(excluded)

    return [word for word in word_split if word not in stop_words]


def transform(title, stopword=True):
    title = title.strip()
    title = lemmatize(title)
    if stopword:
        title = remove_stopword(title)
    return ' '.join(title)

if False:
    # https://openreview-py.readthedocs.io/en/latest/
    # https://readthedocs.org/projects/openreview-py-dm-branch/downloads/pdf/latest/
    # https://docs.openreview.net/getting-started/using-the-api/installing-and-instantiating-the-python-client

    client = openreview.Client(baseurl='https://api.openreview.net')
    # Find invitation ID by running
    # print(client.get_group(id='venues').members) # NeurIPS.cc/2022/Track/Datasets_and_Benchmarks/-/Submission
    # submissions = client.get_all_notes(invitation="ICLR.cc/2024/Conference/-/Blind_Submission", details='directReplies')
    client = openreview.Client(baseurl='https://api2.openreview.net')
    submissions = client.get_all_notes(invitation=f"ICLR.cc/{year}/Conference/-/Submission", details='directReplies')
    # import pdb; pdb.set_trace()
    listofreviewers = set()
else:
    import pickle
    with open("haha.pkl","rb") as fo:
        submissions = pickle.load(fo)
papers = {}

for submission in submissions:
    reviews = []
    for review in submission.details['directReplies']:
        if 'rating' in review['content'].keys():
            rating = int(review['content']['rating']['value'].split(':')[0])
            confidence = int(review['content']['confidence']['value'].split(':')[0])
            aTup = rating,confidence
            reviews.append(aTup)
    papers[submission.content['title']["value"]] = reviews

keywords_all = defaultdict(lambda :0)
for submission in submissions:
    for words in submission.content["keywords"]["value"]:
        keywords_all[words]+=1
# max = 0
# max_key = None
# for key,value in keywords_all.items():
#     if value> max:
#         max_key = key
#         max = value

title_= [submissions[i].content["title"]["value"] for i in range(len(submissions))]
key_words_ = [submissions[i].content["keywords"]["value"] for i in range(len(submissions))]
ratings =[]
for i in range(len(submissions)):
    # try:
    rating = []
    for j in range(len(submissions[i].details["directReplies"])):
        try:
            rating.append(int(submissions[i].details["directReplies"][j]["content"]["rating"]["value"].split(':')[0]))
        except:
            pass
    # rating = [int(submissions[i].details["directReplies"][j]["content"]["rating"]["value"].split(':')[0]) for j in range(len(submissions[i].details["directReplies"]))]
    rating_normalized = np.array(rating).mean()
    ratings.append(rating_normalized)
    # except:
    #     rating
ahaha = pd.DataFrame(np.column_stack([title_,key_words_,ratings]),columns=["title","keywords","rating"])
ahaha = ahaha.sort_values(by='rating', ascending=False)



words = pd.Series(
    ' '.join(ahaha['title'].dropna().apply(transform)).split(' ')
).str.strip()
counts = words.value_counts().sort_values(ascending=True)
plt.subplots(dpi=300)
counts.iloc[-50:].plot.barh(figsize=(8, 12), fontsize=15)
plt.title(f'50 MOST APPEARED TITLE KEYWORDS ({year})', loc='center', fontsize='25',
          fontweight='bold', color='black')
plt.savefig(f'sources/50_most_title_{year}.png', dpi=300, bbox_inches='tight')

ahaha['keywords'] = [",".join(ahaha['keywords'].dropna()[i]) for i in range(len(ahaha["keywords"]))]
words = pd.Series(
    ', '.join(ahaha["keywords"].dropna().apply(partial(transform, stopword=False))).lower().replace(' learn', ' learning').split(',')
).str.strip()
print(words)
counts = words.value_counts().sort_values(ascending=True)
plt.subplots(dpi=300)
counts.iloc[-50:].plot.barh(figsize=(8, 12), fontsize=15)
plt.title(f'50 MOST APPEARED KEYWORDS ({year})', loc='center', fontsize='25',
          fontweight='bold', color='black')
plt.savefig(f'sources/50_most_keywords_{year}.png', dpi=300, bbox_inches='tight')

df_md = ahaha.to_markdown()
with open('output.md', 'w') as file:
    file.write(df_md)








#########################################
# Stats Calculation
allRatings = []
allRatingsMeans = []
for paper in papers:
    ratingsList = []
    for pair in papers[paper]:
        # print(pair[0])
        ratingsList.append(pair[0])

    #paper specific statistics
    mean = np.nanmean(ratingsList)
    median = np.nanmedian(ratingsList)
    stdev = np.nanstd(ratingsList)

    statsTup = mean, median, stdev
    papers[paper].insert(0, statsTup)
    allRatingsMeans.append(mean)

overallMean = np.nanmean(allRatingsMeans)
overallMedian = np.nanmedian(allRatingsMeans)
overallStdev = np.nanstd(allRatingsMeans)
print(f"Mean, Mean Paper Rating: {overallMean}")
print(f"Median, Mean Paper Rating: {overallMedian}")
print(f"Standard Deviation of Mean Paper Rating: {overallStdev}")
print(f"Total Papers: {len(allRatingsMeans)}")
print(f"Total Papers with Nan Reviews: {np.sum(np.isnan(allRatingsMeans))}")



# Write to spreadsheet
if not os.path.exists("csvs"):
    os.makedirs("csvs")
handle = open(f"csvs/openreview_ratings_{datetime.today().strftime('%Y_%m_%d')}.csv", 'w')
handle.write("Title,Mean,Median,StDev," + "Rating,Confidence,"*8 + "\n")
for title in papers:
    stuff = str(papers[title])
    stuff = stuff.replace("(","").replace(")","").replace("[","").replace("]","")
    handle.write(title.replace("\n","").replace(",","") + "," + stuff + "\n")
handle.close()


# Visualize results
fig, ax = plt.subplots(figsize=(20,10))

counts, bins, patches = ax.hist(allRatingsMeans, bins=50)
# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)
# Set the xaxis's tick labels to be formatted with 1 decimal place...
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

# Label the raw counts and the percentages below the x-axis...
bin_centers = 0.5 * np.diff(bins) + bins[:-1]
for count, x in zip(counts, bin_centers):
    # Label the raw counts
    ax.annotate(str(int(count)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -18), textcoords='offset points', va='top', ha='center')

    # Label the percentages
    percent = '%0.0f%%' % (100 * float(count) / counts.sum())
    ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -32), textcoords='offset points', va='top', ha='center')

if not os.path.exists("figs"):
    os.makedirs("figs")
plt.savefig(f"figs/hist_{datetime.today().strftime('%Y_%m_%d')}.png")
plt.show(block=True)
