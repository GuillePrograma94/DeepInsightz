
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load




def recomienda_tfid(new_basket):

    cestas = pd.read_csv('cestas.csv') 
    productos = pd.read_csv('productos.csv')

    # Cargar la matriz TF-IDF y el modelo
    tfidf_matrix = load('tfidf_matrix.joblib') 
    # MAtriz que tienen cada columna los diferentes artículos y las diferentes cestas en las filas
    # Los valores son la importancia de cada artículo en la cesta según las veces que aparece en la misma y el total de artículos
    tfidf = load('tfidf_model.joblib')

    # Convertir la nueva cesta en formato TF-IDF
    new_basket_str = ' '.join(new_basket)
    new_basket_tfidf = tfidf.transform([new_basket_str])

    # Comparar la nueva cesta con las anteriores
    # Calculando la distancia coseoidal, distancia entre rectas
    similarities = cosine_similarity(new_basket_tfidf, tfidf_matrix)
    # La similitud coseno devuelve un valor entre 0 y 1, donde 1 significa 
    # que las cestas son idénticas en términos de productos y 0 que no comparten ningún producto.

    # Obtener los índices de las cestas más similares
    # Muestra los índices de Las 3 cestas más parecidas atendiendo a la distancia calculada anteriormente
    similar_indices = similarities.argsort()[0][-4:]  # Las 3 más similares

    # Crear un diccionario para contar las recomendaciones
    recommendations_count = {}
    total_similarity = 0

    # Recomendar productos de cestas similares
    for idx in similar_indices:
        sim_score = similarities[0][idx]
        # sim_score es el valor de similitud de la cesta actual con la cesta similar.
        total_similarity += sim_score # Suma de las similitudes entre 0 y el nº de cestas similares
        products = cestas.iloc[idx]['Cestas'].split()
        
        for product in products:
            if product.strip() not in new_basket:  # Evitar recomendar lo que ya está en la cesta
                recommendations_count[product.strip()] = recommendations_count.get(product.strip(), 0) + sim_score
                #  se utiliza para incrementar el conteo del producto en recommendations_count.
                # almacena el conteo de la relevancia de cada producto basado en cuántas veces aparece en las cestas similares, ponderado por la similitud de cada cesta.
                # sumandole sim_score se incrementa el score cuando la cesta es mas similar

    # Calcular la probabilidad relativa de cada producto recomendado
    recommendations_with_prob = []
    if total_similarity > 0:  # Verificar que total_similarity no sea cero
        recommendations_with_prob = [(product, score / total_similarity) for product, score in recommendations_count.items()]
        # Se guarda cada producto junto su score calculada
    else:
        print("No se encontraron similitudes suficientes para calcular probabilidades.")
     
    recommendations_with_prob.sort(key=lambda x: x[1], reverse=True)  # Ordenar por puntuación

    # Crear un nuevo DataFrame para almacenar las recomendaciones
    recommendations_data = []
    
    for product, score in recommendations_with_prob:
        # Buscar la descripción en el DataFrame de productos
        description = productos.loc[productos['ARTICULO'] == product, 'DESCRIPCION']
        if not description.empty:
            recommendations_data.append({
                'ARTICULO': product,
                'DESCRIPCION': description.values[0],  # Obtener el primer valor encontrado
                'RELEVANCIA': score
            })

    recommendations_df = pd.DataFrame(recommendations_data)
    
    return recommendations_df