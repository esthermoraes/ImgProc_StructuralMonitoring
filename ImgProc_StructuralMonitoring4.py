import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Função para ajustar contraste e brilho
def ajustar_contraste_brilho(imagem, alpha=1.5, beta=30):
    # Alpha controla o contraste, beta controla o brilho
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

# Função para detectar pontos (círculos) na imagem com pré-processamento
def detectar_pontos_circulos(imagem):
    # Converte a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aumenta o contraste e o brilho
    imagem_ajustada = ajustar_contraste_brilho(imagem_cinza)
    
    # Aplica um leve desfoque para reduzir ruído
    imagem_suavizada = cv2.GaussianBlur(imagem_ajustada, (9, 9), 2)
    
    # Aplica detecção de bordas
    bordas = cv2.Canny(imagem_suavizada, 50, 150)
    
    # Detecta círculos usando a Transformada de Hough
    circulos = cv2.HoughCircles(bordas, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                param1=100, param2=15, minRadius=10, maxRadius=50)
    
    coordenadas_pontos = []
    
    # Verifica se algum círculo foi detectado
    if circulos is not None:
        # Converte as coordenadas dos círculos para inteiros
        circulos = np.round(circulos[0, :]).astype("int")
        
        # Itera sobre os círculos detectados
        for (x, y, r) in circulos:
            coordenadas_pontos.append((x, y))
    
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
        plt.plot(x, y, 'gx')  # Plota os pontos na imagem como cruzes verdes ('gx')
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
            # Detecta os pontos (círculos) na imagem
            coordenadas_pontos = detectar_pontos_circulos(imagem)

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