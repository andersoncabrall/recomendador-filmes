import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def carregar_dados(caminho_csv):
    return pd.read_csv(caminho_csv)


def preparar_dados(tabela):
    tabela = tabela.fillna("")
    tabela["features"] = (
        tabela["title"] + " " +
        tabela["director"] + " " +
        tabela["genre"] + " " +
        tabela["keywords"]
    )
    return tabela


def gerar_tfidf(textos):
    vetorizar = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    return vetorizar.fit_transform(textos)


def calcular_similaridade(matriz_tfidf):
    return cosine_similarity(matriz_tfidf)


def listar_filmes(tabela):
    for titulo in tabela["title"]:
        print("-", titulo)


def obter_indice_filme(tabela):
    nome = input("\nQual filme você assistiu? ").strip().lower()
    filtro = tabela["title"].str.lower() == nome
    if not filtro.any():
        print("Filme não encontrado.")
        return None
    return tabela[filtro].index[0]


def usuario_gostou():
    return input("Você gostou do filme? (s/n): ").strip().lower() == "s"


def recomendar(indice_filme, matriz_similaridade, gostou):
    pontuacoes = list(enumerate(matriz_similaridade[indice_filme]))
    pontuacoes.sort(key=lambda x: x[1], reverse=gostou)
    return pontuacoes[1:4]


def mostrar_recomendacoes(tabela, recomendacoes):
    print("\nRecomendações:\n")
    for indice, _ in recomendacoes:
        print(f"- {tabela.iloc[indice]['title']} ({tabela.iloc[indice]['director']})")


def main():
    tabela = carregar_dados("filmes_recomendador.csv")
    tabela = preparar_dados(tabela)

    matriz_tfidf = gerar_tfidf(tabela["features"])
    matriz_similaridade = calcular_similaridade(matriz_tfidf)

    listar_filmes(tabela)

    indice_filme = obter_indice_filme(tabela)
    if indice_filme is None:
        return

    gostou = usuario_gostou()
    recomendacoes = recomendar(indice_filme, matriz_similaridade, gostou)

    mostrar_recomendacoes(tabela, recomendacoes)


main()
