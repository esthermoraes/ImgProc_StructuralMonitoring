import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Função para detectar os pontos na imagem e retornar as coordenadas dos centros
def detectar_pontos(imagem):
    # Converte a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplica um leve desfoque para suavizar a imagem
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    
    # Aplica o método de Canny para detectar bordas na imagem
    bordas = cv2.Canny(imagem_suavizada, 50, 150)
    
    # Encontra os contornos na imagem
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    coordenadas_pontos = []
    
    # Itera sobre os contornos detectados
    for contorno in contornos:
        # Calcula a área do contorno para filtrar ruído
        area = cv2.contourArea(contorno)
        
        if area > 30:  # Filtra contornos muito pequenos
            # Calcula o centro de massa do contorno
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                coordenadas_pontos.append((cX, cY))
    
    return coordenadas_pontos

# Função para salvar os dados de coordenadas em um arquivo CSV
def salvar_dados(coordenadas_pontos):
    # Cria um DataFrame com as coordenadas dos pontos
    df_pontos = pd.DataFrame(coordenadas_pontos, columns=["X", "Y"])
    
    # Salva os dados em um arquivo CSV
    df_pontos.to_csv("coordenadas_pontos.csv", index=False)
    
    print("Coordenadas salvas em 'coordenadas_pontos.csv'")
    print(df_pontos)

# Função para exibir a imagem e os pontos detectados
def exibir_imagem_com_pontos(imagem, coordenadas_pontos):
    # Exibe a imagem com os pontos detectados
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))  # Converte de BGR para RGB para exibição correta
    for (x, y) in coordenadas_pontos:
        plt.plot(x, y, 'gx')  # Plota os pontos na imagem como cruzes verdes('gx')
    plt.title("Pontos Detectados")
    plt.show()

if __name__ == "__main__":
    # Abre uma janela para selecionar o arquivo de imagem
    Tk().withdraw()  # Oculta a janela principal do Tkinter
    caminho_imagem = askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.jpg;*.jpeg;*.png;*.bmp")])

    if caminho_imagem:
        # Carrega a imagem selecionada
        imagem = cv2.imread(caminho_imagem)
        
        if imagem is not None:
            # Detecta os pontos na imagem
            coordenadas_pontos = detectar_pontos(imagem)

            # Exibe as coordenadas dos pontos detectados
            print("Coordenadas dos pontos detectados:")
            for i, coord in enumerate(coordenadas_pontos):
                print(f"Ponto {i + 1}: X = {coord[0]}, Y = {coord[1]}")

            # Salva os dados em um arquivo CSV
            salvar_dados(coordenadas_pontos)

            # Exibe a imagem com os pontos destacados
            exibir_imagem_com_pontos(imagem, coordenadas_pontos)
        else:
            print("Erro ao carregar a imagem.")
    else:
        print("Nenhuma imagem selecionada.")