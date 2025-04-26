import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import math

# Função para ajustar contraste e brilho
def ajustar_contraste_brilho(imagem, alpha=1.2, beta=20):
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

# Função para melhorar a imagem
def melhorar_imagem(imagem_cinza):
    imagem_equalizada = cv2.equalizeHist(imagem_cinza)
    imagem_ajustada = ajustar_contraste_brilho(imagem_equalizada, alpha=1.5, beta=30)
    imagem_suavizada = cv2.GaussianBlur(imagem_ajustada, (5, 5), 0)
    return imagem_suavizada

# Função para detectar bordas
def detectar_bordas(imagem):
    return cv2.Canny(imagem, threshold1=50, threshold2=150)

# Função para detectar cantos com filtros avançados
def detectar_cantos_com_filtro(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_melhorada = melhorar_imagem(imagem_cinza)
    bordas = detectar_bordas(imagem_melhorada)
    mascara = cv2.dilate(bordas, None, iterations=2)
    
    pontos = cv2.goodFeaturesToTrack(
        bordas,
        maxCorners=1000,          # Limitar número de cantos detectados
        qualityLevel=0.1,       # Tornar mais seletivo
        minDistance=20,         # Distância mínima entre pontos
        mask=mascara            # Usar máscara para restringir áreas de interesse
    )
    
    if pontos is None:
        print("Nenhum ponto detectado após aplicação do filtro.")
        return []
    
    pontos = np.int32(pontos)
    return [(p[0][0], p[0][1]) for p in pontos]

# Função para filtrar pontos muito próximos
def filtrar_pontos(coordenadas_pontos, limiar_distancia=15):
    pontos_filtrados = []
    for ponto in coordenadas_pontos:
        if not pontos_filtrados:
            pontos_filtrados.append(ponto)
        elif all(calcular_distancia(ponto, p) > limiar_distancia for p in pontos_filtrados):
            pontos_filtrados.append(ponto)
    return pontos_filtrados

# Função para calcular distância euclidiana entre dois pontos
def calcular_distancia(pontoA, pontoB):
    return math.sqrt((pontoB[0] - pontoA[0]) ** 2 + (pontoB[1] - pontoA[1]) ** 2)

# Função para calcular distâncias a partir do primeiro ponto
def calcular_distancias_a_partir_do_inicial(coordenadas_pontos):
    distancias = []
    if coordenadas_pontos:
        ponto_inicial = coordenadas_pontos[0]
        for i, ponto in enumerate(coordenadas_pontos):
            if i == 0:
                continue
            distancia = calcular_distancia(ponto_inicial, ponto)
            distancias.append((1, i + 1, distancia))
    return distancias

# Função para salvar coordenadas em CSV
def salvar_coordenadas(coordenadas_pontos, caminho_imagem):
    pasta_csv = "csv_coordenadas"
    os.makedirs(pasta_csv, exist_ok=True)
    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]
    nome_csv = os.path.join(pasta_csv, f"{nome_arquivo}_coordenadas.csv")
    df_pontos = pd.DataFrame(coordenadas_pontos, columns=["X", "Y"])
    df_pontos.to_csv(nome_csv, index=False)
    print(f"Coordenadas salvas em '{nome_csv}'")

# Função para salvar distâncias em CSV
def salvar_distancias(distancias, caminho_imagem):
    pasta_csv = "csv_distancias"
    os.makedirs(pasta_csv, exist_ok=True)
    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]
    nome_csv = os.path.join(pasta_csv, f"{nome_arquivo}_distancias.csv")
    df_distancias = pd.DataFrame(distancias, columns=["Ponto A", "Ponto B", "Distância"])
    df_distancias.to_csv(nome_csv, index=False)
    print(f"Distâncias salvas em '{nome_csv}'")

# Função para exibir imagem com pontos detectados
def exibir_imagem_com_pontos(imagem, coordenadas_pontos):
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    for (x, y) in coordenadas_pontos:
        plt.plot(x, y, 'gx')
    plt.gca().invert_yaxis()
    plt.title("Pontos Detectados")
    plt.show()

# Função principal
def main():
    Tk().withdraw()
    caminho_imagem = askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.jpg;*.jpeg;*.png;*.bmp")])

    if caminho_imagem:
        print(f"Caminho da imagem selecionada: {caminho_imagem}")
        
        # Tentar ler a imagem com um caminho de arquivo absoluto
        imagem = cv2.imread(caminho_imagem)
        
        if imagem is None:
            print("Erro ao carregar a imagem. Verifique se o caminho está correto ou se a imagem está corrompida.")
            return

        # Se a imagem for carregada corretamente, prossiga com o processamento
        coordenadas_pontos = detectar_cantos_com_filtro(imagem)
        coordenadas_pontos = filtrar_pontos(coordenadas_pontos, limiar_distancia=20)

        if not coordenadas_pontos:
            print("Não foi possível detectar pontos na imagem.")
            return
        
        print("Coordenadas dos pontos detectados:")
        for i, coord in enumerate(coordenadas_pontos):
            print(f"Ponto {i + 1}: X = {coord[0]}, Y = {coord[1]}")

        salvar_coordenadas(coordenadas_pontos, caminho_imagem)

        distancias = calcular_distancias_a_partir_do_inicial(coordenadas_pontos)
        print("\nDistâncias a partir do ponto inicial:")
        for dist in distancias:
            print(f"Distância entre Ponto {dist[0]} e Ponto {dist[1]}: {dist[2]:.2f}")

        salvar_distancias(distancias, caminho_imagem)
        exibir_imagem_com_pontos(imagem, coordenadas_pontos)
    else:
        print("Nenhuma imagem selecionada.")

if __name__ == "__main__":
    main()