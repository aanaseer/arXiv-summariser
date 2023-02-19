import arxiv
import cohere


api_key = "6YjoNayGiocXhyVjG9zGovJfdgbdDbrX8PZKmcxE"
co = cohere.Client(api_key)

def generate_summary(abstract, temperature=0.5):
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
    search = arxiv.Search(
        query = keyword,
        max_results = num_articles,
        sort_by = arxiv.SortCriterion.SubmittedDate
        )
    results = [result for result in search.results()]
    return results


def get_summary_and_misc_info(search_results, temperature):
    titles, authors, article_links, summaries = [], [], [], []
    for i, result in enumerate(search_results):
        title = result.title
        authors = [author.name for author in result.authors]
        article_link = result.pdf_url

        summary = generate_summary(result.summary, temperature=temperature)
        titles.append(title)
        authors.append(authors)
        article_links.append(article_link)
        summaries.append(summary)

    return titles, authors, article_links, summaries



