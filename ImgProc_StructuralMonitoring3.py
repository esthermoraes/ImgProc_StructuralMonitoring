import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Função para carregar uma imagem e convertê-la em escala de cinza
def carregar_imagem(caminho_imagem):
    img = cv2.imread(caminho_imagem)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# Função para detectar bordas (possível detecção de fissuras em estruturas)
def detectar_bordas(img_gray):
    edges = cv2.Canny(img_gray, 50, 150)
    return edges

# Função para processar os dados e calcular áreas de fissuras
def calcular_areas_fissuras(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    return areas

# Função para exibir imagem e os resultados
def exibir_resultados(img, edges):
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Original')
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Bordas Detectadas')
    plt.show()

# Função principal para realizar o processamento de uma imagem
def processar_imagem(caminho_imagem):
    img, img_gray = carregar_imagem(caminho_imagem)
    edges = detectar_bordas(img_gray)
    areas_fissuras = calcular_areas_fissuras(edges)
    
    # Exibindo os resultados
    exibir_resultados(img, edges)
    
    # Armazenando resultados em um DataFrame Pandas
    df_areas = pd.DataFrame(areas_fissuras, columns=['Área da Fissura'])
    
    # Estatísticas das áreas de fissuras
    estatisticas = df_areas.describe()
    print(estatisticas)
    
    # Visualizando dados
    plt.hist(df_areas['Área da Fissura'], bins=10, color='blue', alpha=0.7)
    plt.title('Distribuição das Áreas das Fissuras')
    plt.xlabel('Área da Fissura')
    plt.ylabel('Frequência')
    plt.show()

# Exemplo de uso
caminho_imagem = 'images.jpg'
processar_imagem(caminho_imagem)