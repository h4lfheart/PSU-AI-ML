import json

import numpy as np
from openai import OpenAI

# no leaking key!!!
with open("token.secret", "r") as token_file:
    client = OpenAI(api_key=token_file.read())

with open("articles.json", "r") as articles_file:
    articles: dict[str, str] = json.loads(articles_file.read())

def get_embedding(query):
    return client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

article_embeddings = list(map(get_embedding, articles.values()))

def get_article(query):

    query_embedding = get_embedding(query)

    similarities = [cosine_similarity(query_embedding, article_embedding) for article_embedding in article_embeddings]

    return list(articles.items())[similarities.index(max(similarities))]


use_recap = input("Would you like to use Chat-GPT recap mode? [y/n]\n").lower() in ["y", "yes", "yuh", "yur", "yessir", "bet", "yeah"]

while True:
    question = input("What do you need help with? (Ex. How do i find my blender installation?)\n")
    if len(question) == 0 or all(map(lambda char: char in ["", " "], question)):
        print("Question cannot be empty! How else am I supposed to help smh\n")
        continue

    title, content = get_article(question)

    if use_recap:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who is recapping articles to help the user. Answer with this with strict context, following original wording strictly, be very direct."},
                {
                    "role": "user",
                    "content": f"User question: {question}. The content is {content}.",
                }
            ]
        )

        print(response.choices[0].message.content)
    else:
        print(f"{title}:")
        print("---------------------------------------")
        print(content)
        print("\n")


