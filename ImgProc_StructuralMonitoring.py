# Importando as bibliotecas necessárias
import cv2             # OpenCV para processamento de imagem
import numpy as np      # NumPy para manipulação de arrays
import pandas as pd     # Pandas para manipulação de dados
import matplotlib.pyplot as plt  # Matplotlib para visualização de dados

# Função para capturar imagem de uma câmera ou vídeo
def capturar_imagem(fonte_video=0):
    cap = cv2.VideoCapture(fonte_video)
    
    if not cap.isOpened():
        print("Erro ao acessar a câmera ou vídeo.")
        return None

    # Capturar uma única imagem (frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Erro ao capturar a imagem.")
        return None

    # Exibir a imagem capturada
    cv2.imshow("Imagem Capturada", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return frame

# Função para pré-processar a imagem (conversão para escala de cinza e redução de ruído)
def preprocessamento_imagem(imagem):
    # Conversão para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicação de filtro para redução de ruído
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    
    return imagem_suavizada

# Função para capturar imagem de uma câmera ou vídeo
def capturar_imagem(fonte_video=0):
    cap = cv2.VideoCapture(fonte_video)
    
    if not cap.isOpened():
        print("Erro ao acessar a câmera ou vídeo.")
        return None

    # Capturar uma única imagem (frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Erro ao capturar a imagem.")
        return None

    # Exibir a imagem capturada
    cv2.imshow("Imagem Capturada", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return frame

# Função para pré-processar a imagem (conversão para escala de cinza e redução de ruído)
def preprocessamento_imagem(imagem):
    # Conversão para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # Aplicação de filtro para redução de ruído
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    
    return imagem_suavizada

# Função para detectar fissuras usando o método de Canny
def detectar_fissuras(imagem_preprocessada):
    # Aplicando o algoritmo de Canny para detecção de bordas
    bordas = cv2.Canny(imagem_preprocessada, 100, 200)

    # Exibir as fissuras detectadas
    plt.figure(figsize=(10, 6))
    plt.title("Fissuras Detectadas")
    plt.imshow(bordas, cmap='gray')
    plt.show()

    return bordas

# Exemplo de função para salvar dados de fissuras detectadas
def salvar_dados_fissuras(bordas):
    # Inicializa um DataFrame com as coordenadas dos pixels que representam fissuras
    fissuras_coords = np.column_stack(np.where(bordas > 0))  # Obtendo as coordenadas das fissuras
    
    # Criação de um DataFrame para armazenar as coordenadas das fissuras
    df_fissuras = pd.DataFrame(fissuras_coords, columns=["Y", "X"])

    # Exibir o início dos dados para verificação
    print(df_fissuras.head())

    # Salvar os dados em um arquivo CSV
    df_fissuras.to_csv("dados_fissuras.csv", index=False)

    return df_fissuras

if __name__ == "__main__":
    # Captura a imagem
    imagem = capturar_imagem()
    
    if imagem is not None:
        # Pré-processa a imagem
        imagem_preprocessada = preprocessamento_imagem(imagem)

        # Detecta fissuras
        bordas_fissuras = detectar_fissuras(imagem_preprocessada)

        # Salva os dados das fissuras detectadas
        salvar_dados_fissuras(bordas_fissuras)

"""
Próximos Passos:
Refinamento da detecção de fissuras: Implementar algoritmos para melhorar a detecção e reduzir falsos positivos.
Análise contínua: Aplicar o monitoramento ao longo do tempo para detectar evolução das fissuras.
Integração com sensores adicionais (se necessário): Comparar os dados obtidos via imagem com sensores físicos para validação.
"""