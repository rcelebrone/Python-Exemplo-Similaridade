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

# Exemplo de uso:
texto_a = "O gato pulou sobre o telhado."
texto_b = "O cachorro correu no telhado."
texto_c = "O gato preto."

sim_ab = calcular_similaridade_textos(texto_a, texto_b)
sim_ac = calcular_similaridade_textos(texto_a, texto_c)

print(f"Similaridade entre '{texto_a}' e '{texto_b}': {sim_ab:.2f}")
print(f"Similaridade entre '{texto_a}' e '{texto_c}': {sim_ac:.2f}")
