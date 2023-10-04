# Libreries
from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
import imutils
import mediapipe as mp
import math
import os
import face_recognition as fr
from ultralytics import YOLO

# Face Code
def Code_Face(images):
    listacod = []

    # Iteramos
    for img in images:
        # Correccion de color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Codificamos la imagen
        cod = fr.face_encodings(img)[0]
        # Almacenamos
        listacod.append(cod)

    return listacod

# Close Windows LogBiometric
def Close_Windows():
    global step, conteo
    # Reset Variables
    conteo = 0
    step = 0
    pantalla2.destroy()

# Close Windows SignBiometric
def Close_Windows2():
    global step, conteo
    # Reset Variables
    conteo = 0
    step = 0
    pantalla3.destroy()

# Object Detection
def Object_Detection(img):
    global glass, capHat
    glass = False
    capHat = False

    # img
    frame = img

    # Clases
    clsNameCap = ['Gafas', 'Sombrero', 'Abrigo', 'Camisa', 'Pantalones', 'Shorts', 'Falda', 'Vestido', 'Maleta','Zapato']
    clsNameGlass = ['Gafas']

    # Cap & Glass Detect
    resultsCap = modelCap(frame, stream=True, imgsz=640)
    resultsGlass = modelGlass(frame, stream=True, imgsz=640)

    # Cap
    for resCap in resultsCap:
        # Boxes Cap
        boxesCap = resCap.boxes
        for boxCap in boxesCap:
            # Bounding box
            xi1, yi1, xf1, yf1 = boxCap.xyxy[0]
            xi1, yi1, xf1, yf1 = int(xi1), int(yi1), int(xf1), int(yf1)

            # Error < 0
            if xi1 < 0: xi1 = 0
            if yi1 < 0: yi1 = 0
            if xf1 < 0: xf1 = 0
            if yf1 < 0: yf1 = 0

            # Class
            clsCap = int(boxCap.cls[0])

            # Confidence
            confCap = math.ceil(boxCap.conf[0])

            if clsCap == 1:
                capHat = True
                # Draw Cap
                cv2.rectangle(frame, (xi1, yi1), (xf1, yf1), (255, 255, 0), 2)
                cv2.putText(frame, f"{clsNameCap[clsCap]} {int(confCap * 100)}%", (xi1, yi1 - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    # Glass
    for resGlass in resultsGlass:
        # Boxes Glass
        boxesGlass = resGlass.boxes
        for boxGlass in boxesGlass:
            # Bounding box
            xi2, yi2, xf2, yf2 = boxGlass.xyxy[0]
            xi2, yi2, xf2, yf2 = int(xi2), int(yi2), int(xf2), int(yf2)

            # Error < 0
            if xi2 < 0: xi2 = 0
            if yi2 < 0: yi2 = 0
            if xf2 < 0: xf2 = 0
            if yf2 < 0: yf2 = 0

            # Class
            clsGlass = int(boxGlass.cls[0])

            # Confidence
            confGlass = math.ceil(boxGlass.conf[0])

            if clsGlass == 0:
                glass = True
                # Draw Cap
                cv2.rectangle(frame, (xi2, yi2), (xf2, yf2), (255, 0, 255), 2)
                cv2.putText(frame, f"{clsNameGlass[clsGlass]} {int(confGlass * 100)}%", (xi2, yi2 - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    return frame


# Profile
def Profile():
    global step, conteo, UserName, OutFolderPathUser
    # Reset Variables
    conteo = 0
    step = 0

    pantalla4 = Toplevel(pantalla)
    pantalla4.title("BIOMETRIC SIGN")
    pantalla4.geometry("1280x720")

    back = Label(pantalla4, image=imagenB, text="Back")
    back.place(x=0, y=0, relwidth=1, relheight=1)

    # Archivo
    UserFile = open(f"{OutFolderPathUser}/{UserName}.txt", 'r')
    InfoUser = UserFile.read().split(',')
    Name = InfoUser[0]
    User = InfoUser[1]
    Pass = InfoUser[2]
    UserFile.close()

    # Check
    if User in clases:
        # Interfaz
        texto1 = Label(pantalla4, text=f"BIENVENIDO {Name}")
        texto1.place(x=580, y=50)
        # Label
        # Video
        lblImgUser = Label(pantalla4)
        lblImgUser.place(x=490, y=80)

        # Imagen
        PosUserImg = clases.index(User)
        UserImg = images[PosUserImg]

        ImgUser = Image.fromarray(UserImg)
        #
        ImgUser = cv2.imread(f"{OutFolderPathFace}/{User}.png")
        ImgUser = cv2.cvtColor(ImgUser, cv2.COLOR_RGB2BGR)
        ImgUser = Image.fromarray(ImgUser)
        #
        IMG = ImageTk.PhotoImage(image=ImgUser)

        lblImgUser.configure(image=IMG)
        lblImgUser.image = IMG


# Register Biometric
def Log_Biometric():
    global pantalla, pantalla2, conteo, parpadeo, img_info, step, glass, capHat

    # Leemos la videocaptura
    if cap is not None:
        ret, frame = cap.read()

        # Frame Save
        frameSave = frame.copy()

        # RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameObject = frame.copy()

        # Si es correcta
        if ret == True:

            # Inference
            res = FaceMesh.process(frameRGB)

            # Object Detect
            frame = Object_Detection(frameObject)

            # List Results
            px = []
            py = []
            lista = []
            r = 5
            t = 3

            # Resultados
            if res.multi_face_landmarks:
                # Iteramos
                for rostros in res.multi_face_landmarks:

                    # Draw Face Mesh
                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject.FACE_CONNECTIONS, ConfigDraw, ConfigDraw)

                    # Extract KeyPoints
                    for id, puntos in enumerate(rostros.landmark):

                        # Info IMG
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        # 468 KeyPoints
                        if len(lista) == 468:
                            # Ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            longitud1 = math.hypot(x2 - x1, y2 - y1)
                            #print(longitud1)

                            # Ojo Izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                            longitud2 = math.hypot(x4 - x3, y4 - y3)
                            #print(longitud2)

                            # Parietal Derecho
                            x5, y5 = lista[139][1:]
                            # Parietal Izquierdo
                            x6, y6 = lista[368][1:]

                            # Ceja Derecha
                            x7, y7 = lista[70][1:]
                            # Ceja Izquierda
                            x8, y8 = lista[300][1:]

                            # Face Detect
                            faces = detector.process(frameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:

                                    # bboxInfo - "id","bbox","score","center"
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # Threshold
                                    if score > confThreshold:
                                        # Info IMG
                                        alimg, animg, c = frame.shape

                                        # Coordenates
                                        xi, yi, an, al = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, an, al = int(xi * animg), int(yi * alimg), int(
                                            an * animg), int(al * alimg)

                                        # Width
                                        offsetan = (offsetx / 100) * an
                                        xi = int(xi - int(offsetan/2))
                                        an = int(an + offsetan)
                                        xf = xi + an

                                        # Height
                                        offsetal = (offsety / 100) * al
                                        yi = int(yi - offsetal)
                                        al = int(al + offsetal)
                                        yf = yi + al

                                        # Error < 0
                                        if xi < 0: xi = 0
                                        if yi < 0: yi = 0
                                        if an < 0: an = 0
                                        if al < 0: al = 0

                                    # Steps
                                    if step == 0 and glass == False and capHat == False:
                                        # Draw
                                        cv2.rectangle(frame, (xi, yi, an, al), (255, 0, 255), 2)
                                        # IMG Step0
                                        alis0, anis0, c = img_step0.shape
                                        frame[50:50 + alis0, 50:50 + anis0] = img_step0

                                        # IMG Step1
                                        alis1, anis1, c = img_step1.shape
                                        frame[50:50 + alis1, 1030:1030 + anis1] = img_step1

                                        #IMG Step2
                                        alis2, anis2, c = img_step2.shape
                                        frame[270:270 + alis2, 1030:1030 + anis2] = img_step2

                                        # Condiciones
                                        if x7 > x5 and x8 < x6:

                                            # Cont Parpadeos
                                            if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:  # Parpadeo
                                                conteo = conteo + 1
                                                parpadeo = True

                                            elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:  # Seguridad parpadeo
                                                parpadeo = False

                                            # IMG check
                                            alich, anich, c = img_check.shape
                                            frame[165:165 + alich, 1105:1105 + anich] = img_check

                                            # Parpadeos
                                            # Conteo de parpadeos
                                            cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375), cv2.FONT_HERSHEY_COMPLEX,0.5,
                                                        (255, 255, 255), 1)


                                            if conteo >= 3:
                                                # IMG check
                                                alich, anich, c = img_check.shape
                                                frame[385:385 + alich, 1105:1105 + anich] = img_check

                                                # Open Eyes
                                                if longitud1 > 14 and longitud2 > 14:
                                                    # Cut
                                                    cut = frameSave[yi:yf, xi:xf]
                                                    # Save Image Without Draw
                                                    cv2.imwrite(f"{OutFolderPathFace}/{RegUser}.png", cut)
                                                    # Cerramos
                                                    step = 1
                                        else:
                                            conteo = 0

                                    if step == 1 and glass == False and capHat == False:
                                        # Draw
                                        cv2.rectangle(frame, (xi, yi, an, al), (0, 255, 0), 2)
                                        # IMG check Liveness
                                        allich, anlich, c = img_liche.shape
                                        frame[50:50 + allich, 50:50 + anlich] = img_liche

                                    if glass == True:
                                        # IMG Glass
                                        algla, angla, c = img_glass.shape
                                        frame[50:50 + algla, 50:50 + angla] = img_glass

                                    if capHat == True:
                                        # IMG CapHat
                                        alcap, ancap, c = img_cap.shape
                                        frame[50:50 + alcap, 50:50 + ancap] = img_cap

                            # Close Window
                            close = pantalla2.protocol("WM_DELETE_WINDOW", Close_Windows)

            # Rendimensionamos el video
            frame = imutils.resize(frame, width=1280)

            # Convertimos el video
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            # Mostramos en el GUI
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Log_Biometric)

        else:
            cap.release()

# Sign Biometric
def Sign_Biometric():
    global pantalla, pantalla3, conteo, parpadeo, img_info, step, UserName, prueba

    # Leemos la videocaptura
    if cap is not None:
        ret, frame = cap.read()

        # Frame Save
        frameCopy = frame.copy()

        # Resize
        frameFR = cv2.resize(frameCopy, (0, 0), None, 0.25, 0.25)

        # Color
        rgb = cv2.cvtColor(frameFR, cv2.COLOR_BGR2RGB)

        # RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameObject = frame.copy()

        # Si es correcta
        if ret == True:

            # Inference
            res = FaceMesh.process(frameRGB)
            # Object Detect
            frame = Object_Detection(frameObject)

            # List Results
            px = []
            py = []
            lista = []
            r = 5
            t = 3

            # Resultados
            if res.multi_face_landmarks:
                # Iteramos
                for rostros in res.multi_face_landmarks:

                    # Draw Face Mesh
                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject.FACE_CONNECTIONS, ConfigDraw, ConfigDraw)

                    # Extract KeyPoints
                    for id, puntos in enumerate(rostros.landmark):

                        # Info IMG
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        # 468 KeyPoints
                        if len(lista) == 468:
                            # Ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            longitud1 = math.hypot(x2 - x1, y2 - y1)
                            #print(longitud1)

                            # Ojo Izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                            longitud2 = math.hypot(x4 - x3, y4 - y3)
                            #print(longitud2)

                            # Parietal Derecho
                            x5, y5 = lista[139][1:]
                            # Parietal Izquierdo
                            x6, y6 = lista[368][1:]

                            # Ceja Derecha
                            x7, y7 = lista[70][1:]
                            # Ceja Izquierda
                            x8, y8 = lista[300][1:]

                            # Face Detect
                            faces = detector.process(frameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:

                                    # bboxInfo - "id","bbox","score","center"
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # Threshold
                                    if score > confThreshold:
                                        # Info IMG
                                        alimg, animg, c = frame.shape

                                        # Coordenates
                                        xi, yi, an, al = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, an, al = int(xi * animg), int(yi * alimg), int(
                                            an * animg), int(al * alimg)

                                        # Width
                                        offsetan = (offsetx / 100) * an
                                        xi = int(xi - int(offsetan/2))
                                        an = int(an + offsetan)
                                        xf = xi + an

                                        # Height
                                        offsetal = (offsety / 100) * al
                                        yi = int(yi - offsetal)
                                        al = int(al + offsetal)
                                        yf = yi + al

                                        # Error < 0
                                        if xi < 0: xi = 0
                                        if yi < 0: yi = 0
                                        if an < 0: an = 0
                                        if al < 0: al = 0

                                        # Steps
                                        if step == 0 and glass == False and capHat == False:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, an, al), (255, 0, 255), 2)
                                            # IMG Step0
                                            alis0, anis0, c = img_step0.shape
                                            frame[50:50 + alis0, 50:50 + anis0] = img_step0

                                            # IMG Step1
                                            alis1, anis1, c = img_step1.shape
                                            frame[50:50 + alis1, 1030:1030 + anis1] = img_step1

                                            # IMG Step2
                                            alis2, anis2, c = img_step2.shape
                                            frame[270:270 + alis2, 1030:1030 + anis2] = img_step2

                                            # Condiciones
                                            if x7 > x5 and x8 < x6:

                                                # Cont Parpadeos
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:  # Parpadeo
                                                    conteo = conteo + 1
                                                    parpadeo = True

                                                elif longitud2 > 10 and longitud2 > 10 and parpadeo == True:  # Seguridad parpadeo
                                                    parpadeo = False

                                                # IMG check
                                                alich, anich, c = img_check.shape
                                                frame[165:165 + alich, 1105:1105 + anich] = img_check

                                                # Parpadeos
                                                # Conteo de parpadeos
                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375),
                                                            cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                                            (255, 255, 255), 1)

                                                if conteo >= 3:
                                                    # IMG check
                                                    alich, anich, c = img_check.shape
                                                    frame[385:385 + alich, 1105:1105 + anich] = img_check

                                                    # Open Eyes
                                                    if longitud1 > 14 and longitud2 > 14:
                                                        step = 1
                                            else:
                                                conteo = 0

                                        if step == 1 and glass == False and capHat == False:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, an, al), (0, 255, 0), 2)
                                            # IMG check Liveness
                                            allich, anlich, c = img_liche.shape
                                            frame[50:50 + allich, 50:50 + anlich] = img_liche

                                            # Find Faces
                                            faces = fr.face_locations(rgb)
                                            facescod = fr.face_encodings(rgb, faces)

                                            # Iteramos
                                            for facecod, faceloc in zip(facescod, faces):

                                                # Matching
                                                Match = fr.compare_faces(FaceCode, facecod)

                                                # Similitud
                                                simi = fr.face_distance(FaceCode, facecod)

                                                # Min
                                                min = np.argmin(simi)

                                                # User
                                                if Match[min]:
                                                    # UserName
                                                    UserName = clases[min].upper()

                                                    Profile()
                                        if glass == True:
                                            # IMG Glass
                                            algla, angla, c = img_glass.shape
                                            frame[50:50 + algla, 50:50 + angla] = img_glass

                                        if capHat == True:
                                            # IMG CapHat
                                            alcap, ancap, c = img_cap.shape
                                            frame[50:50 + alcap, 50:50 + ancap] = img_cap


                            # Close Window
                            close = pantalla3.protocol("WM_DELETE_WINDOW", Close_Windows2)

            # Rendimensionamos el video
            frame = imutils.resize(frame, width=1280)

            # Convertimos el video
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            # Mostramos en el GUI
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Sign_Biometric)

        else:
            cap.release()

# Sign Function
def Sign():
    global LogUser, LogPass, OutFolderPath, cap, lblVideo, pantalla3, FaceCode, clases, images

    # DB Faces
    # Accedemos a la carpeta
    images = []
    clases = []
    lista = os.listdir(OutFolderPathFace)

    # Leemos los rostros del DB
    for lis in lista:
        # Leemos las imagenes de los rostros
        imgdb = cv2.imread(f'{OutFolderPathFace}/{lis}')
        # Almacenamos imagen
        images.append(imgdb)
        # Almacenamos nombre
        clases.append(os.path.splitext(lis)[0])

    # Face Code
    FaceCode = Code_Face(images)

    # 3° Ventana
    pantalla3 = Toplevel(pantalla)
    pantalla3.title("BIOMETRIC SIGN")
    pantalla3.geometry("1280x720")

    back2 = Label(pantalla3, image=imagenB, text="Back")
    back2.place(x=0, y=0, relwidth=1, relheight=1)

    # Video
    lblVideo = Label(pantalla3)
    lblVideo.place(x=0, y=0)

    # Elegimos la camara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    Sign_Biometric()


# Register Function
def Log():
    global RegName, RegUser, RegPass, InputNameReg, InputUserReg, InputPassReg, cap, lblVideo, pantalla2
    # Name, User, PassWord
    RegName, RegUser, RegPass = InputNameReg.get(), InputUserReg.get(), InputPassReg.get()

    if len(RegName) == 0 or len(RegUser) == 0 or len(RegPass) == 0:
        # Info incompleted
        print(" FORMULARIO INCOMPLETO ")

    else:
        # Info Completed
        # Check users
        UserList = os.listdir(PathUserCheck)
        # Name Users
        UserName = []
        for lis in UserList:
            # Extract User
            User = lis
            User = User.split('.')
            # Save
            UserName.append(User[0])

        # Check Names
        if RegUser in UserName:
            # Registred
            print("USUARIO REGISTRADO ANTERIORMENTE")

        else:
            # No Registred
            # Info
            info.append(RegName)
            info.append(RegUser)
            info.append(RegPass)

            # Save Info
            f = open(f"{OutFolderPathUser}/{RegUser}.txt", 'w')
            f.writelines(RegName + ',')
            f.writelines(RegUser + ',')
            f.writelines(RegPass + ',')
            f.close()

            # Clean
            InputNameReg.delete(0, END)
            InputUserReg.delete(0, END)
            InputPassReg.delete(0, END)

            # Ventana principal
            pantalla2 = Toplevel(pantalla)
            pantalla2.title("BIOMETRIC REGISTER")
            pantalla2.geometry("1280x720")

            back = Label(pantalla2, image=imagenB, text="Back")
            back.place(x=0, y=0, relwidth=1, relheight=1)

            # Video
            lblVideo = Label(pantalla2)
            lblVideo.place(x=0, y=0)

            # Elegimos la camara
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(3, 1280)
            cap.set(4, 720)
            Log_Biometric()

# Confidence
confidenceCap = 0.5
confidenceGlass = 0.5
# Umbral
confThresholdCap = 0.5
confThresholdGlass = 0.5

# Modelo
modelGlass = YOLO("Modelos/Gafas.pt")
modelCap = YOLO("Modelos/Gorras.pt")

# Path
OutFolderPathUser = 'C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/DataBase/Users'
PathUserCheck = 'C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/DataBase/Users/'
OutFolderPathFace = 'C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/DataBase/Faces'

# List
info = []

# Variables
parpadeo = False
conteo = 0
muestra = 0
step = 0

# Margen
offsety = 30
offsetx = 20

# Umbral
confThreshold = 0.5
blurThreshold = 15

# Tool Draw
mpDraw = mp.solutions.drawing_utils
ConfigDraw = mpDraw.DrawingSpec(thickness=1, circle_radius=1) #Ajustamos la configuracion de dibujo

# Object Face Mesh
FacemeshObject = mp.solutions.face_mesh
FaceMesh = FacemeshObject.FaceMesh(max_num_faces=1)

# Object Detect
FaceObject = mp.solutions.face_detection
detector = FaceObject.FaceDetection(min_detection_confidence= 0.5, model_selection=1)

# Img OpenCV
# Leer imagenes
img_cap = cv2.imread("C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/cap.png")
img_glass = cv2.imread("C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/glass.png")
img_check = cv2.imread("C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/check.png")
img_step0 = cv2.imread("C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/Step0.png")
img_step1 = cv2.imread("C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/Step1.png")
img_step2 = cv2.imread("C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/Step2.png")
img_liche = cv2.imread("C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/LivenessCheck.png")

# Ventana principal
pantalla = Tk()
pantalla.title("FACE RECOGNITION SYSTEM")
pantalla.geometry("1280x720")

# Fondo
imagenF = PhotoImage(file="C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/Inicio.png")
background = Label(image = imagenF, text = "Inicio")
background.place(x = 0, y = 0, relwidth = 1, relheight = 1)

# Fondo 2
imagenB = PhotoImage(file="C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/Back2.png")

# Input Text
# Register
# Name
InputNameReg = Entry(pantalla)
InputNameReg.place(x= 110, y = 320)
# User
InputUserReg = Entry(pantalla)
InputUserReg.place(x= 110, y = 430)
# Pass
InputPassReg = Entry(pantalla)
InputPassReg.place(x= 110, y = 540)

# Botones
# Registro
imagenBR = PhotoImage(file="C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/BtSign.png")
BtReg = Button(pantalla, text="Registro", image=imagenBR, height="40", width="200", command=Log)
BtReg.place(x = 300, y = 580)

# Inicio de sesion
imagenBL = PhotoImage(file="C:/Users/santi/OneDrive/Desktop/YouTube/Programacion/Vision Python/FaceRecognitionLivenessSystem/SetUp/BtLogin.png")
BtSign = Button(pantalla, text="Sign", image=imagenBL, height="40", width="200", command=Sign)
BtSign.place(x = 850, y = 580)

# Eject
pantalla.mainloop()