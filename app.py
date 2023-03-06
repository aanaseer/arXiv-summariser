"""This is the main file for the app. It contains the Flask app and the functions that generate the summary and the arxiv search results."""

import os

import arxiv
import cohere

from collections import defaultdict

from flask import Flask, render_template, request


api_key = os.getenv("CO_KEY")
co = cohere.Client(api_key)

def generate_summary(abstract, temperature=0.2):
    """Returns a summary of the abstract."""
    base_idea_prompt = """Passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn't the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to "the dusty section of the dictionary" to find its latest words.
                        TLDR: Wordle has not gotten more difficult to solve.
                        --
                        Passage: ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised $190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, and Plato Capital.
                        TLDR: ArtificialIvan has raised $190 million in Series C funding.
                        --"""
    abstract = abstract.replace("\n"," ")
    response = co.generate(
        model='xlarge',
        prompt = base_idea_prompt + "\nPassage: " + abstract + "\nTLDR: ",
        max_tokens=55,
        temperature= temperature,
        k=0,
        p=0.7,
        frequency_penalty=0.1,
        presence_penalty=0,
        stop_sequences=["--"])

    summary = response.generations[0].text
    summary = summary.split(".")[0] +"."

    return summary


def get_arxiv_search_results(keyword, num_articles):
    """Returns a list of arxiv search results."""
    search = arxiv.Search(
        query = keyword,
        max_results = num_articles,
        sort_by = arxiv.SortCriterion.SubmittedDate
        )
    results = [result for result in search.results()]
    return results


def get_summary_and_misc_info(search_results, temperature):
    """Returns a dictionary with the title of the article as the key and a dictionary with the authors, article link, and summary as the value."""
    summary_and_misc_info = defaultdict()
    for i, result in enumerate(search_results):
        title = result.title
        authors = [author.name for author in result.authors]
        article_link = result.pdf_url
        cohere_summary = generate_summary(result.summary, temperature=temperature)
        
        summary_and_misc_info[title] = {
            "authors": authors,
            "article_link": article_link,
            "summary": cohere_summary
        }
    return summary_and_misc_info

app = Flask(__name__)

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        keywords = request.form["keyword"]
        arxiv_search_results = get_arxiv_search_results(keywords,5)
        summary_info_dict = get_summary_and_misc_info(arxiv_search_results, 0.5)
        return render_template("index.html", summary_info_dict=summary_info_dict)
        
    result = request.args.get("result")
    return render_template("index.html", result=result)

