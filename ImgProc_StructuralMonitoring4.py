import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# Função para ajustar contraste e brilho
def ajustar_contraste_brilho(imagem, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

# Função para detectar pontos (círculos) na imagem com pré-processamento
def detectar_pontos_circulos(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_ajustada = ajustar_contraste_brilho(imagem_cinza)
    imagem_suavizada = cv2.GaussianBlur(imagem_ajustada, (9, 9), 2)
    bordas = cv2.Canny(imagem_suavizada, 50, 150)
    
    circulos = cv2.HoughCircles(bordas, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                param1=100, param2=15, minRadius=10, maxRadius=50)
    
    coordenadas_pontos = []
    if circulos is not None:
        circulos = np.round(circulos[0, :]).astype("int")
        for (x, y, r) in circulos:
            coordenadas_pontos.append((x, y))
    
    return coordenadas_pontos

# Função para salvar os dados de coordenadas em um arquivo CSV dentro da pasta "CSV"
def salvar_dados(coordenadas_pontos, caminho_imagem):
    pasta_csv = "csv"
    os.makedirs(pasta_csv, exist_ok=True)  # Cria a pasta se não existir
    
    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]  # Pega o nome da imagem sem extensão
    nome_csv = os.path.join(pasta_csv, f"{nome_arquivo}_coordenadas.csv")  # Cria o caminho completo do CSV
    
    df_pontos = pd.DataFrame(coordenadas_pontos, columns=["X", "Y"])
    df_pontos.to_csv(nome_csv, index=False)
    
    print(f"Coordenadas salvas em '{nome_csv}'")
    print(df_pontos)

# Função para exibir a imagem e os pontos detectados
def exibir_imagem_com_pontos(imagem, coordenadas_pontos):
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    for (x, y) in coordenadas_pontos:
        plt.plot(x, y, 'gx')  
    plt.title("Pontos Detectados")
    plt.show()

if __name__ == "__main__":
    Tk().withdraw()
    caminho_imagem = askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.jpg;*.jpeg;*.png;*.bmp")])

    if caminho_imagem:
        imagem = cv2.imread(caminho_imagem)
        
        if imagem is not None:
            coordenadas_pontos = detectar_pontos_circulos(imagem)

            print("Coordenadas dos pontos detectados:")
            for i, coord in enumerate(coordenadas_pontos):
                print(f"Ponto {i + 1}: X = {coord[0]}, Y = {coord[1]}")

            salvar_dados(coordenadas_pontos, caminho_imagem)
            exibir_imagem_com_pontos(imagem, coordenadas_pontos)
        else:
            print("Erro ao carregar a imagem.")
    else:
        print("Nenhuma imagem selecionada.")