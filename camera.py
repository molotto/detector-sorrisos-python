import cv2
import dlib
import os

# Verificar se o arquivo do preditor existe
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"Arquivo {predictor_path} não encontrado. Baixe de: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

# Carregar o detector de faces do dlib
detector_faces = dlib.get_frontal_face_detector()
# Carregar o preditor facial do dlib
preditor_facial = dlib.shape_predictor(predictor_path)

# Inicializar a captura de vídeo
video = cv2.VideoCapture(0)

if not video.isOpened():
    raise RuntimeError("Erro ao acessar a câmera.")

while True:
    # Capturar um frame de vídeo
    ret, frame = video.read()
    if not ret:
        print("Erro ao capturar o frame. Encerrando...")
        break

    # Converter o frame para escala de cinza
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no frame
    faces = detector_faces(frame_cinza)

    for face in faces:
        # Detectar pontos faciais na face
        pontos_faciais = preditor_facial(frame_cinza, face)

        # Extrair as coordenadas dos pontos faciais para detectar um sorriso
        canto_esquerdo_boca = (pontos_faciais.part(48).x, pontos_faciais.part(48).y)
        canto_direito_boca = (pontos_faciais.part(54).x, pontos_faciais.part(54).y)
        centro_boca = (pontos_faciais.part(66).x, pontos_faciais.part(66).y)  # Melhor ponto para altura

        # Calcular a largura e altura da boca
        largura_boca = abs(canto_direito_boca[0] - canto_esquerdo_boca[0])
        altura_boca = abs(centro_boca[1] - (canto_esquerdo_boca[1] + canto_direito_boca[1]) // 2)

        # Ajuste na detecção do sorriso
        if largura_boca > 2.0 * altura_boca:  # Ajuste fino na relação largura/altura
            cv2.putText(frame, 'Sorriso detectado!', (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Desenhar um retângulo ao redor da face
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

    # Mostrar o frame com as detecções
    cv2.imshow('Detecção de Sorriso', frame)

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
video.release()
cv2.destroyAllWindows()
