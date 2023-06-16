import cv2

# Carregar o classificador pré-treinado para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Abrir o vídeo de entrada
video_capture = cv2.VideoCapture('arsene.mp4')

# Obter as propriedades do vídeo de entrada
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Criar o objeto para salvar o vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

while video_capture.isOpened():
    # Ler um frame do vídeo
    ret, frame = video_capture.read()
    if not ret:
        break

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhar retângulos ao redor de cada rosto encontrado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibir o frame com as faces identificadas
    cv2.imshow('Video', frame)
    
    # Salva o frame com as faces identificadas no vídeo de saída
    output_video.write(frame)

    # aperte q para sair 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
output_video.release()
cv2.destroyAllWindows()
