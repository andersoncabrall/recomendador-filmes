## 🎬 Recomendador de Filmes com Machine Learning

````md

Sistema simples de recomendação de filmes feito em **Python**, usando técnicas básicas de **Machine Learning**.

---

## ⚙️ Requisitos

Para rodar o projeto, você precisa ter:

- **Python 3.8 ou superior**
- As seguintes bibliotecas instaladas:

```bash
pip install pandas scikit-learn
````

---

## 📁 Estrutura do projeto

```
.
├── recomendador.py
├── filmes_recomendador.csv
└── README.md
```

---

## ▶️ Como rodar o projeto

1. Certifique-se de que o arquivo CSV está na mesma pasta do código:

```
filmes_recomendador.csv
```

2. No terminal, navegue até a pasta do projeto e execute:

```bash
python recomendador.py
```

3. Escolha um filme da lista e responda se gostou ou não para receber as recomendações.

---

## 🧠 Como funciona

* O programa lê os dados do CSV
* Junta título, diretor, gênero e palavras-chave
* Usa **TF-IDF** para transformar texto em números
* Calcula a similaridade entre os filmes
* Recomenda 3 filmes com base na sua escolha

---

## 💡 Tecnologias usadas

* Python
* Pandas
* Scikit-learn

---

Projeto simples, direto e focado em código limpo.

```
```
