import typer
import pandas as pd
from index import BM25Matrix, BertIndex, NavecIndex, Word2vecIndex
from loader import load_navec, load_spacy, load_w2v, load_bert
from rich import print
from rich.console import Console
from rich.table import Table



app = typer.Typer(rich_markup_mode="rich")
console = Console()


@app.command('download_models')
def download_models():
    """
    [bold green]Automatically loads[/bold green] necessary models to RAM. No user activity is needed here.
    """
    navec = load_navec()
    w2v = load_w2v()
    tokenizer, bert = load_bert()
    nlp = load_spacy()

    return navec, w2v, nlp, tokenizer, bert


@app.command()
def search(index: str = typer.Option(default='bm25', help='Eats index name', prompt='Please, specify index name: '),
           query: str = typer.Argument(default=..., help='The query to base the search on. If your query is longer than 1 word, please, use quoates like this \'QUERY\'.'),
           number: int = typer.Option(default=5, help='The number of texts you will receive.')):
    """
    [bold green]Finds texts[/bold green] based on your [i]query[/i]. :sparkles:

    This requires a [underline]query[/underline] and [underline]index name: bm25, navec, w2v, bert[/underline] after [bold]--index[/bold] paparameter.
    [bold]EXAMPLE:[/bold] [green]"__main__.py search --index INDEX_NAME  --n  NUMBER QUERY"[/green] or [green] "__main__.py search QUERY"[/green].
    (In the second option you will be asked to specify the name of index later.)
    """
    df = pd.read_csv('isdb_hw2.csv')
    navec, w2v, nlp, tokenizer, bert = download_models()
    if index == 'bm25':
        index_obj = BM25Matrix(df.clean, df.text)
    elif index == 'navec':
        index_obj = NavecIndex(df.clean, df.text, navec, nlp)
    elif index == 'bert':
        index_obj = BertIndex(df.text, tokenizer, bert)
    elif index == 'w2v':
        index_obj = Word2vecIndex(df.clean, df.text, w2v, nlp)
    else:
        print(f'[bold red]No such index: {index}.[/bold red] Restart the code and try one of the following: bm25, navec, w2v.')
        return

    print('[bold green]Results based on you query: [/bold green]')
    search_result = index_obj.search(query, n=number)
    if -1 in search_result:
        finals = ['Navec does not contain words from your query. Try another index or '
                                                'change your query.']
    elif -2 in search_result:
        finals = ['Your query is semantically empty. Please change your query to '
                                                'proceed searching.']
    else:
        finals = [df.text[id] for id in search_result]
    print(*finals, sep='\n\n')


@app.command('stats')
def get_stats(filename: str = typer.Argument(default='stats.csv',
                                             help='Filename of the table with statistics from the current folder')):
    """
    [bold green]Provides[/bold green] such statistics as [underline]speed[/underline] and [underline]allocated memory[/underline].
    """
    df = pd.read_csv(filename).reset_index(names=['metric'])
    table = Table(title='Statistics')
    table.add_column('Metric', justify='center')
    table.add_column('Time (s)', justify='center')
    table.add_column('Memory (b)', justify='center')
    for idx, row in df.iterrows():
        table.add_row(row['Unnamed: 0'], str(row['time']), str(row['memory']))
    console.print(table)


if __name__ == '__main__':
    app()
