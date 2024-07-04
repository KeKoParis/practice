import spacy

import pandas as pd

nlp = spacy.load("ru_core_news_md")

df = pd.read_excel("data.xlsx")

for i in range(len(df.values)):
    phrase = df.iloc[i].tolist()[0]

    doc = nlp(phrase)

    lemmatized_tokens = [token.lemma_ for token in doc]

    lemmatized_text = " ".join(lemmatized_tokens)

    df.at[i, "data"] = lemmatized_text
    print(df.values[i])

df.to_csv("dataset_1.csv")
