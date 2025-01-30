import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

# Ajustar contraste e brilho
def ajustar_contraste_brilho(imagem, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

# Processar a imagem para detectar pontos escuros e claros
def processar_imagem(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_ajustada = ajustar_contraste_brilho(imagem_cinza)

    # Aplicar um filtro Gaussiano para suavizar ruídos
    imagem_suavizada = cv2.GaussianBlur(imagem_ajustada, (5, 5), 0)

    # Aplicar um detector de bordas para realçar pontos destacados
    bordas = cv2.Canny(imagem_suavizada, 50, 150)

    return bordas

# Detectar cantos na imagem processada
def detectar_cantos(imagem):
    imagem_processada = processar_imagem(imagem)

    maxCorners = 100  # Aumentado para detectar mais pontos
    qualityLevel = 0.01  # Reduzido para detectar mais pontos fracos
    minDistance = 5  # Reduzido para detectar pontos próximos

    # Detectar pontos de interesse
    pontos = cv2.goodFeaturesToTrack(imagem_processada, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)

    if pontos is not None:
        pontos = pontos.astype(int)
        return [(p[0][0], p[0][1]) for p in pontos]
    else:
        return []

# Salvar pontos em CSV
def salvar_dados(coordenadas_pontos, caminho_imagem):
    pasta_csv = "csv_coordenadas"
    os.makedirs(pasta_csv, exist_ok=True)
    
    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]
    nome_csv = os.path.join(pasta_csv, f"{nome_arquivo}_coordenadas.csv")
    
    df_pontos = pd.DataFrame(coordenadas_pontos, columns=["X", "Y"])
    df_pontos.to_csv(nome_csv, index=False)
    
    print(f"Coordenadas salvas em '{nome_csv}'")

# Exibir imagem com pontos
def exibir_imagem_com_pontos(imagem, coordenadas_pontos):
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    for (x, y) in coordenadas_pontos:
        plt.plot(x, y, 'gx', markersize=5)
    plt.gca().invert_yaxis()
    plt.title(f"Pontos Detectados ({len(coordenadas_pontos)})")
    plt.show()

def main():
    Tk().withdraw()
    caminho_imagem = askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.jpg;*.jpeg;*.png;*.bmp")])

    if caminho_imagem:
        imagem = cv2.imread(caminho_imagem)
        
        if imagem is not None:
            coordenadas_pontos = detectar_cantos(imagem)

            print(f"Total de pontos detectados: {len(coordenadas_pontos)}")
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