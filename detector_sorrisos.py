import cv2
import dlib
import os

#^ verifica se o arquivo do preditor existe
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"Arquivo {predictor_path} não encontrado. Baixe de: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

#^ carrega o detector de faces e o preditor facial do dlib
detector_faces = dlib.get_frontal_face_detector()
preditor_facial = dlib.shape_predictor(predictor_path)

#^ liga a câmera
video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError("Erro ao acessar a câmera.")

#^ variável para contar os frames
contador_sorriso = 0

#^ loop para capturar frames
while True:
    ret, frame = video.read()
    if not ret:
        print("Erro ao capturar o frame. Encerrando...")
        break

    #^ converte o frame para escala de cinza
    #* o detector de faces geralmente funciona melhor em imagens em escala de cinza
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #^ detecta faces no frame
    faces = detector_faces(frame_cinza, 1)

    for face in faces:
        pontos_faciais = preditor_facial(frame_cinza, face) #^ detecta os pontos na face

        #^ coordenadas dos pontos faciais para detectar um sorriso
        canto_esquerdo_boca = (pontos_faciais.part(48).x, pontos_faciais.part(48).y)  #* 48 e 54 são os cantos esquerdo e direito
        canto_direito_boca = (pontos_faciais.part(54).x, pontos_faciais.part(54).y)
        centro_boca = (pontos_faciais.part(66).x, pontos_faciais.part(66).y)  #^ melhor ponto para altura, ponto 66 é o centro da boca

        #^ calcula medidas da boca
        largura_boca = abs(canto_esquerdo_boca[0] - canto_direito_boca[0])
        altura_boca = abs(centro_boca[1] - (canto_esquerdo_boca[1] + canto_direito_boca[1]) // 2)
        altura_face = abs(face.bottom() - face.top())  #^ tamanho da face
        taxa_sorriso = largura_boca / altura_face  #^ normaliza pelo tamanho do rosto

        # print(f"Largura boca: {largura_boca}, Altura boca: {altura_boca}, Taxa sorriso: {taxa_sorriso}")

        #^ verifica se a proporçãod a largura da boca em relação à altura da face é maior que 0,5, o que indica um sorriso+ 
        if taxa_sorriso > 0.5 and largura_boca > 3.5 * altura_boca: #^ + juntamente, verifica se a boca está suficientemente aberta
            contador_sorriso += 1
        else:
            contador_sorriso = max(0, contador_sorriso - 1)

        #^ o sorriso precisa aparecer em 4 frames seguidos para ser detectado
        if contador_sorriso > 4:
            cv2.putText(frame, 'Sorriso detectado!', (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        #^ Desenhar um retângulo ao redor da face
        # cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

    #^ mostra o frame com as detecções
    cv2.imshow('image', frame)

    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
        break
    
video.release()
cv2.destroyAllWindows()
