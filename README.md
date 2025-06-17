# Python Exemplo Similaridade

Este é um projeto simples em Python que mostra como calcular a similaridade entre dois textos usando Similaridade de Cosseno e a técnica Bag-of-Words.

🚀 Como FuncionaO script compara textos contando as palavras que eles têm em comum. 
- Quanto mais palavras em comum (ou palavras semelhantes em frequência), maior a similaridade.

⚙️ ConfiguraçãoPara usar este código, você precisa ter Python e a biblioteca scikit-learn instalados.

🐍 1. Instalar Python (no Ubuntu)Atualize e instale:
- sudo apt update
- sudo apt install python3 python3-pip
- Verifique:python3 --version
- pip3 --version

🧠 2. Instalar scikit-learnInstale a biblioteca:pip3 install scikit-learn
- Verifique (opcional):
- Abra o python3 e digite import sklearn; print(sklearn.__version__).

📦 3. Ambientes Virtuais (Opcional, mas recomendado)
- Crie um ambiente virtual para organizar suas dependências:
- Crie: python3 -m venv venvAtive: source venv/bin/activate
- Instale scikit-learn (dentro do venv): pip install scikit-learn
- Execute o script no terminal:python3 main.py

📝 Código
```
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calcular_similaridade_textos(texto1, texto2):
    # Pré-processamento simples: converter para minúsculas
    texto1 = texto1.lower()
    texto2 = texto2.lower()

    textos = [texto1, texto2]

    # Criar vetores de frequência de palavras (Bag-of-Words)
    vectorizer = CountVectorizer()
    vetores = vectorizer.fit_transform(textos)

    # Calcular a similaridade de cosseno
    # A similaridade será uma matriz 2x2, pegamos o valor [0,1] ou [1,0]
    similaridade = cosine_similarity(vetores[0], vetores[1])

    return similaridade[0][0]
```
## Exemplo de uso:
texto_a = "O gato pulou sobre o telhado."
texto_b = "O cachorro correu no telhado."
texto_c = "O gato preto."

sim_ab = calcular_similaridade_textos(texto_a, texto_b)
sim_ac = calcular_similaridade_textos(texto_a, texto_c)

print(f"Similaridade entre '{texto_a}' e '{texto_b}': {sim_ab:.2f}")
print(f"Similaridade entre '{texto_a}' e '{texto_c}': {sim_ac:.2f}")
