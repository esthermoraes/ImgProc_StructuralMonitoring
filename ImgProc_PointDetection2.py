import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
from sklearn.cluster import DBSCAN  # Para remover pontos muito próximos

# Função para ajustar contraste e brilho
def ajustar_contraste_brilho(imagem, alpha=1.5, beta=30):
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

# Função para remover pontos duplicados usando DBSCAN (agrupamento baseado em densidade)
def remover_pontos_duplicados(pontos, eps=5, min_samples=1):
    if not pontos:
        return []
    
    pontos_np = np.array(pontos)  # Converte para numpy array
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pontos_np)
    
    # Para cada cluster, pegamos o ponto central
    pontos_filtrados = []
    for label in set(clustering.labels_):
        if label != -1:
            cluster_pontos = pontos_np[clustering.labels_ == label]
            media_x, media_y = np.mean(cluster_pontos, axis=0)
            pontos_filtrados.append((int(media_x), int(media_y)))

    return pontos_filtrados

# Função para detectar os cantos (features) da imagem
def detectar_cantos(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_ajustada = ajustar_contraste_brilho(imagem_cinza)
    imagem_suavizada = cv2.GaussianBlur(imagem_ajustada, (5, 5), 0)
    
    # Usar goodFeaturesToTrack para detectar cantos sem limite de pontos
    pontos = cv2.goodFeaturesToTrack(imagem_suavizada, maxCorners=0, qualityLevel=0.01, minDistance=5)
    
    if pontos is not None:
        pontos = [tuple(p.ravel()) for p in np.int0(pontos)]  # Converte para tuplas para evitar variações de valores
        pontos_filtrados = remover_pontos_duplicados(pontos, eps=5)  # Filtra pontos muito próximos
        return pontos_filtrados
    else:
        return []

# Função para salvar os dados de coordenadas em um arquivo CSV
def salvar_dados(coordenadas_pontos, caminho_imagem):
    pasta_csv = "csv"
    os.makedirs(pasta_csv, exist_ok=True)  # Cria a pasta se não existir
    
    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]  # Nome da imagem sem extensão
    nome_csv = os.path.join(pasta_csv, f"{nome_arquivo}_coordenadas.csv")  # Caminho do CSV
    
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
    caminho_imagem = askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.jpg;*.jpeg;*.png;*.bmp")])

    if caminho_imagem:
        imagem = cv2.imread(caminho_imagem)
        
        if imagem is not None:
            coordenadas_pontos = detectar_cantos(imagem)

            if coordenadas_pontos:
                print("Coordenadas dos pontos detectados:")
                for i, coord in enumerate(coordenadas_pontos):
                    print(f"Ponto {i + 1}: X = {coord[0]}, Y = {coord[1]}")

                salvar_dados(coordenadas_pontos, caminho_imagem)
                exibir_imagem_com_pontos(imagem, coordenadas_pontos)
            else:
                print("Nenhum ponto detectado.")
            
        else:
            print("Erro ao carregar a imagem.")
    else:
        print("Nenhuma imagem selecionada.")

if __name__ == "__main__":
    main()