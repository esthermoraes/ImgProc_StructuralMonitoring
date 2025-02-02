import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
from itertools import combinations
import math

# Ajustar contraste e brilho
def ajustar_contraste_brilho(imagem, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

# Converter cinza para preto
def converter_cinza_para_preto(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_cinza[imagem_cinza > 180] = 0  # Transforma em preto diretamente
    return imagem_cinza

# Detectar cantos na imagem
def detectar_cantos(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_ajustada = ajustar_contraste_brilho(imagem_cinza)
    imagem_suavizada = cv2.GaussianBlur(imagem_ajustada, (5, 5), 0)
    
    # Ajustando parâmetros para detecção de pontos
    maxCorners = 0  # Limite máximo de pontos a serem detectados
    qualityLevel = 0.05  # Aumentado para filtrar pontos de qualidade mais baixa
    minDistance = 15  # Distância mínima entre pontos

    # Detecta pontos com goodFeaturesToTrack
    pontos = cv2.goodFeaturesToTrack(imagem_suavizada, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)

    if pontos is not None:
        pontos = pontos.astype(int)

        # Lista para armazenar os pontos filtrados
        coordenadas_pontos = []

        for p in pontos:
            x, y = p[0]
            # Verificar se o ponto é suficientemente distante dos outros pontos
            is_distante = True
            for (px, py) in coordenadas_pontos:
                if np.linalg.norm(np.array([x, y]) - np.array([px, py])) < minDistance:
                    is_distante = False
                    break
            if is_distante:
                coordenadas_pontos.append((x, y))

        return coordenadas_pontos
    else:
        return []

# Salvar pontos em CSV
def salvar_coordenadas(coordenadas_pontos, caminho_imagem):
    pasta_csv = "csv_coordinates"
    os.makedirs(pasta_csv, exist_ok=True)
    
    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]
    nome_csv = os.path.join(pasta_csv, f"{nome_arquivo}_coordinates.csv")
    
    df_pontos = pd.DataFrame(coordenadas_pontos, columns=["X", "Y"])
    df_pontos.to_csv(nome_csv, index=False)
    
    print(f"Coordenadas salvas em '{nome_csv}'")

# Função para calcular a distância euclidiana entre dois pontos
def calcular_distancia(pontoA, pontoB):
    return math.sqrt((pontoB[0] - pontoA[0]) ** 2 + (pontoB[1] - pontoA[1]) ** 2)

# Função para calcular as distâncias entre todos os pontos
def calcular_distancias(coordenadas_pontos):
    distancias = []
    for (i, pontoA), (j, pontoB) in combinations(enumerate(coordenadas_pontos), 2):
        distancia = calcular_distancia(pontoA, pontoB)
        distancias.append((i + 1, j + 1, distancia))  # Adiciona índices dos pontos e a distância
    return distancias

# Função para salvar as distâncias em um arquivo CSV dentro da pasta "CSV"
def salvar_distancias(distancias, caminho_imagem):
    pasta_csv = "csv_distances"
    os.makedirs(pasta_csv, exist_ok=True)  # Cria a pasta se não existir
    
    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]  # Pega o nome da imagem sem extensão
    nome_csv = os.path.join(pasta_csv, f"{nome_arquivo}_distances.csv")  # Cria o caminho completo do CSV
    
    df_distancias = pd.DataFrame(distancias, columns=["Ponto A", " Ponto B", " Distância"])
    df_distancias.to_csv(nome_csv, index=False)
    
    print(f"Distâncias salvas em '{nome_csv}'")

def exibir_imagem_com_pontos(imagem, coordenadas_pontos, imagem_cinza):
    # Exibe a imagem original com os pontos
    plt.subplot(1, 2, 1)  # Cria uma subplot para a imagem original
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    for (x, y) in coordenadas_pontos:
        plt.plot(x, y, 'gx')
    plt.gca().invert_yaxis()
    plt.title(f"Imagem Original com Pontos ({len(coordenadas_pontos)})")

    # Exibe a imagem em tons de cinza com os pontos
    plt.subplot(1, 2, 2)  # Cria uma subplot para a imagem em tons de cinza
    plt.imshow(imagem_cinza, cmap='gray')

    plt.gca().invert_yaxis()
    plt.title(f"Imagem em Tons de Cinza com Pontos ({len(coordenadas_pontos)})")

    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
    plt.show()

def main():
    Tk().withdraw()
    caminho_imagem = askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.jpg;*.jpeg;*.png;*.bmp")])

    if caminho_imagem:
        imagem = cv2.imread(caminho_imagem)
        
        if imagem is not None:
            coordenadas_pontos = detectar_cantos(imagem)

            imagem_cinza = converter_cinza_para_preto(imagem)

            print(f"Total de pontos detectados: {len(coordenadas_pontos)}")
            for i, coord in enumerate(coordenadas_pontos):
                print(f"Ponto {i + 1}: X = {coord[0]}, Y = {coord[1]}")

            salvar_coordenadas(coordenadas_pontos, caminho_imagem)

            # Calcular e exibir as distâncias em seguida
            distancias = calcular_distancias(coordenadas_pontos)
            print("\nDistâncias entre os pontos:")
            for dist in distancias:
                print(f"Distância entre Ponto {dist[0]} e Ponto {dist[1]}: {dist[2]:.2f}")

            # Salvar as distâncias
            salvar_distancias(distancias, caminho_imagem)

            # Exibir a imagem com os pontos
            exibir_imagem_com_pontos(imagem, coordenadas_pontos, imagem_cinza)

        else:
            print("Erro ao carregar a imagem.")
            
    else:
        print("Nenhuma imagem selecionada.")

if __name__ == "__main__":
    main()