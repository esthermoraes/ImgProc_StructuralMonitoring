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

# Função para detectar os cantos (features) da imagem
def detectar_cantos(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_ajustada = ajustar_contraste_brilho(imagem_cinza)
    imagem_suavizada = cv2.GaussianBlur(imagem_ajustada, (5, 5), 0)
    
    # Usar goodFeaturesToTrack para detectar cantos
    pontos = cv2.goodFeaturesToTrack(imagem_suavizada, maxCorners=None, qualityLevel=0.001, minDistance=10)
    pontos = np.int0(pontos)
    
    coordenadas_pontos = []
    for p in pontos:
        x, y = p.ravel()
        coordenadas_pontos.append((x, y))
    
    return coordenadas_pontos

# Função para salvar os dados de coordenadas em um arquivo CSV dentro da pasta "CSV"
def salvar_dados(coordenadas_pontos, caminho_imagem):
    pasta_csv = "csv_coordenadas"
    os.makedirs(pasta_csv, exist_ok=True)  # Cria a pasta se não existir
    
    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]  # Pega o nome da imagem sem extensão
    nome_csv = os.path.join(pasta_csv, f"{nome_arquivo}_coordenadas.csv")  # Cria o caminho completo do CSV
    
    df_pontos = pd.DataFrame(coordenadas_pontos, columns=["X", "Y"])
    df_pontos.to_csv(nome_csv, index=False)
    
    print(f"Coordenadas salvas em '{nome_csv}'")

# Função para exibir a imagem e os pontos detectados
def exibir_imagem_com_pontos(imagem, coordenadas_pontos):
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    for (x, y) in coordenadas_pontos:
        plt.plot(x, y, 'gx')
    plt.gca().invert_yaxis()  # Inverter o eixo Y para que o zero fique em baixo
    plt.title("Pontos Detectados")
    plt.show()

def main():
    Tk().withdraw()
    caminho_imagem = askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", ".jpg;.jpeg;.png;.bmp")])

    if caminho_imagem:
        imagem = cv2.imread(caminho_imagem)
        
        if imagem is not None:
            coordenadas_pontos = detectar_cantos(imagem)

            print("Coordenadas dos pontos detectados:")
            for i, coord in enumerate(coordenadas_pontos):
                print(f"Ponto {i + 1}: X = {coord[0]}, Y = {coord[1]}")

            salvar_dados(coordenadas_pontos, caminho_imagem)
            exibir_imagem_com_pontos(imagem, coordenadas_pontos)
        else:
            print("Erro ao carregar a imagem.")
    else:
        print("Nenhuma imagem selecionada.")

if __name__ == "__main__":
    main()