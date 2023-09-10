# ref: https://github.com/yizhongw/self-instruct/blob/main/self_instruct/instruction_visualize.ipynb
# Install benepar parser, refer to https://github.com/nikitakit/self-attentive-parser#installation

import argparse
import os
import numpy as np
import pandas as pd
import json
import tqdm
import plotly.express as px
import spacy
import benepar
import plotly.io as pio
import matplotlib.pyplot as plt
from config_path import get_Sparkles_path

benepar.download('benepar_en3')
nlp = spacy.load('en_core_web_md')
# doc = nlp("The time for action is now. It's never too late to do something.")
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})  # pip install protobuf==3.20.0


def find_root_verb_and_its_dobj(tree_root):
    # first check if the current node and its children satisfy the condition
    if tree_root.pos_ == "VERB":
        for child in tree_root.children:
            if child.dep_ == "dobj" and child.pos_ == "NOUN":
                return tree_root.lemma_, child.lemma_
        return tree_root.lemma_, None
    # if not, check its children
    for child in tree_root.children:
        return find_root_verb_and_its_dobj(child)
    # if no children satisfy the condition, return None
    return None, None


def find_root_verb_and_its_dobj_in_string(s):
    doc = nlp(s)
    first_sent = list(doc.sents)[-1]  # for a user message, usually the last sentence is the question?
    verb, noun = find_root_verb_and_its_dobj(first_sent.root)
    if (verb is None or noun is None) and len(list(doc.sents)) > 1:
        first_sent = list(doc.sents)[0]
        verb2, noun2 = find_root_verb_and_its_dobj(first_sent.root)
        if verb is None and noun is None or verb2 and noun2:
            return verb2, noun2
    return verb, noun


def get_sentences(parsed_results, msgidx=[0,2], imgnum=[1,2,3], subset="VG"):
    sentences = []
    dialogue_indices = []
    for jdx, item in enumerate(parsed_results):
        dialogue = item["dialogue"]
        for idx in msgidx:
            num = dialogue[idx]["content"].count("<Img><ImageHere></Img>")
            if num in imgnum or len(imgnum) == 0:
                sentences.append(dialogue[idx]["content"].replace("<Img><ImageHere></Img>", ""))
                dialogue_indices.append(jdx)
    save_name = f"msgidx{''.join([str(_) for _ in msgidx])}_imgnum{''.join([str(_) for _ in imgnum])}_{subset}"
    return save_name, sentences, dialogue_indices


def analyze_verb_noun_dist(sentences, dialogue_indices, save_dir, save_name):
    data = []
    for instruction, dialogue_idx in tqdm.tqdm(zip(sentences, dialogue_indices)):
        try:
            verb, noun = find_root_verb_and_its_dobj_in_string(instruction)
            data.append({"verb": verb, "noun": noun, "instruction": instruction, "dialogue_idx": dialogue_idx})
        except Exception as e:
            print(e)
            print(instruction)

    df = pd.DataFrame(data)

    # removing any rows with missing values from raw_phrases
    phrases = df.dropna()

    # This groups the DataFrame by verb and noun, counts the number of occurrences of each verb-noun pair,
    # and sorts them in descending order.
    # The result is not assigned to a variable and is therefore not used in the rest of the function.
    # phrases[["verb", "noun"]].groupby(["verb", "noun"]).size().sort_values(ascending=False)

    # This groups the DataFrame by verb, counts the number of occurrences of each verb,
    # selects the 20 most common verbs, and returns a new DataFrame with the verbs and their counts.
    top_verbs = phrases[["verb"]].groupby(["verb"]).size().nlargest(20).reset_index()

    # This filters the phrases DataFrame to only include rows where the verb is one of the top 20 verbs.
    df = phrases[phrases["verb"].isin(top_verbs["verb"].tolist())]

    # This groups the new DataFrame by verb and noun, counts the number of occurrences of each verb-noun pair,
    # converts the result into a DataFrame, renames the count column to "count",
    # and sorts the DataFrame by count in descending order.
    df = df.groupby(["verb", "noun"]).size().reset_index().rename(columns={0: "count"}).sort_values(by=["count"], ascending=False)

    # This groups the DataFrame by verb, sorts each group by count in descending order,
    # selects the top 4 rows from each group, and returns a new DataFrame with the result.
    df = df.groupby("verb").apply(lambda x: x.sort_values("count", ascending=False).head(4)).reset_index(drop=True)

    # This creates a sunburst chart using the DataFrame. The chart has two levels:
    # the first level is the verb and the second level is the noun.
    # The size of each section is determined by the count.
    fig = px.sunburst(df, path=['verb', 'noun'], values='count')

    # This updates the layout of the chart to remove the margins and set the font family and size.
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), font_family="Times New Roman",
                      font_size=28, autosize=False, width=1000, height=1000)

    fig.write_html(f"{save_dir}/verb_noun_dist_{save_name}.html")
    pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469#issuecomment-994907721
    pio.write_image(fig, f"{save_dir}/verb_noun_dist_{save_name}.pdf")
    print(f"Saved verb noun distribution to {save_dir}/verb_noun_dist_{save_name}.pdf")


def plot_length_distribution(sentences, save_dir, save_name, title='Distribution of Sentence Lengths'):
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    sentence_lengths = sorted(sentence_lengths)
    average_length = np.mean(sentence_lengths)
    print(f'Average sentence length: {average_length}')

    # # Square-root rule
    # num_bins = int(np.sqrt(len(sentence_lengths)))

    # Freedman-Diaconis rule
    iqr = np.percentile(sentence_lengths, 75) - np.percentile(sentence_lengths, 25)
    bin_width = 2 * iqr / (len(sentence_lengths) ** (1/3))
    num_bins = int((max(sentence_lengths) - min(sentence_lengths)) / bin_width)

    # If you encounter error of "TypeError: vars() argument must have __dict__ attribute" when using Pycharm,
    # Go to File -> Settings -> Tools -> Python Scientific and uncheck "Show plots in tool window"
    plt.figure(figsize=(10, 6))

    # plt.hist(sentence_lengths, bins=num_bins, color='skyblue', edgecolor='black')
    # 'hist' is a list of frequencies, 'bins' is a list of bin edge values
    hist, bins, _ = plt.hist(sentence_lengths, bins=num_bins, color='skyblue', edgecolor='black')

    plt.xlabel('Sentence Length (words)', fontsize=20)
    plt.ylabel('Number of Sentences', fontsize=20)
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(f"{save_dir}/len_dist_{save_name}.pdf")
    print(f"Sentence length distribution saved to {save_dir}/len_dist_{save_name}.pdf")

    # calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # return x and y points
    points = list(zip(bin_centers, hist))
    return points


def generate_word_cloud(sentences, save_dir, save_name):
    from wordcloud import WordCloud, STOPWORDS

    # Combine all sentences into one large string
    text = ' '.join(sentences)

    # Create a list of stopwords
    stopwords = set(STOPWORDS)

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          max_words=80,
                          max_font_size=150,
                          min_font_size=30,
                          prefer_horizontal=1.0).generate(text)

    # Plot the word cloud
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(f"{save_dir}/word_cloud_{save_name}.png")  # seems not support pdf
    print(f"Word cloud saved to {save_dir}/word_cloud_{save_name}.png")


def group_and_rank_questions_by_verb_noun(sentences, dialogue_indices, root, save_name):
    # `grouped_questions_ranked` is a list of dictionaries, where each dictionary represents a unique question group
    # characterized by a specific verb-noun combination, or lack thereof. Each dictionary contains the verb and noun
    # (which can be None), a list of related instructions along with their corresponding dialogue indices
    # (under the key "instr_dialogue"), and the count of instructions in the group. The list is sorted in descending
    # order of the counts, so groups with more instructions come first.
    res_path = f"{root}/data/grouped_questions_{save_name}.json"
    if os.path.exists(res_path):
        with open(res_path, "r") as f:
            return json.load(f)

    data = []
    for instruction, dialogue_idx in tqdm.tqdm(zip(sentences, dialogue_indices)):
        try:
            verb, noun = find_root_verb_and_its_dobj_in_string(instruction)
            data.append({"verb": verb, "noun": noun, "instruction": instruction, "dialogue_idx": dialogue_idx})
        except Exception as e:
            print(e)
            print(instruction)

    df = pd.DataFrame(data)
    df["instr_dialogue"] = list(zip(df["instruction"], df["dialogue_idx"]))
    df["idx"] = df.index  # Create a unique identifier for each row

    # Create a new column 'group' that is equal to 'idx' when both verb and noun are None, and 'verb_noun' otherwise
    df['group'] = np.where(df['verb'].isnull() & df['noun'].isnull(), df['idx'],
                           df['verb'].astype(str) + '_' + df['noun'].astype(str))

    # Group the DataFrame by 'group'
    groups = df.groupby("group")

    unique_items = []
    for _, group in groups:
        verb, noun = group.iloc[0]["verb"], group.iloc[0]["noun"]  # Get verb and noun from the first row in the group
        instr_dialogue = [{'instruction': i, 'dialogue_idx': d} for i, d in group["instr_dialogue"]]
        count = len(instr_dialogue)
        unique_items.append({"verb": verb, "noun": noun, "instr_dialogue": instr_dialogue, "count": count})

    grouped_questions_ranked = sorted(unique_items, key=lambda x: x['count'], reverse=True)
    # for dialogues whose 1st turn contains two images: 898 items, count 1: 224-897; count 2: 125-223; count 3: 89-124;

    with open(res_path, 'w') as fw:
        json.dump(grouped_questions_ranked, fw)

    return grouped_questions_ranked


def generate_statistics(path, subset):
    with open(path, 'r') as fr:
        parsed_results = json.load(fr)

    # for user messages (questions)
    msgidx = [0, 2]
    imgnum = []
    save_name, sentences, dialogue_indices = get_sentences(parsed_results, msgidx, imgnum, subset=subset)
    title = "Distribution of User Messages' Lengths"
    plot_length_distribution(sentences, save_dir=statistics_dir, save_name=save_name, title=title)
    analyze_verb_noun_dist(sentences, dialogue_indices, save_dir=statistics_dir, save_name=save_name)

    # for assistant messages (answers)
    msgidx = [1, 3]
    imgnum = []
    title = "Distribution of Assistant Messages' Lengths"
    save_name, sentences, _ = get_sentences(parsed_results, msgidx, imgnum, subset=subset)
    plot_length_distribution(sentences, save_dir=statistics_dir, save_name=save_name, title=title)
    generate_word_cloud(sentences, save_dir=statistics_dir, save_name=save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sparkles_root", type=str, default=None, help=(""))
    args = parser.parse_args()

    sparkles_root, SparklesDialogueCC_root, SparklesDialogueVG_root, SparklesDialogueCC_path, \
    SparklesDialogueVG_path, SparklesEval_path, BISON_path, NLVR2_path, statistics_dir = \
        get_Sparkles_path(args.sparkles_root)

    os.makedirs(statistics_dir, exist_ok=True)

    generate_statistics(path=SparklesDialogueVG_path, subset="VG")
    generate_statistics(path=SparklesDialogueCC_path, subset="CC")

    print("Done!")