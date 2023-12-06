import cv2 as cv

path = 'C:/Users/pc/Desktop/DetectionOpenCv/source/lunettes/1.png'

visage_cascade = cv.CascadeClassifier('C:/Users/pc/Desktop/DetectionOpenCv//cascade/visage.xml')
cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv.flip(img, 1)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    visages = visage_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=4)

    for x, y, w, h in visages:
        lunettes_img = cv.imread(path, -1)
        if w * h > 2000:
            l_min = int(y + 0.3 * h)
            l_max = int(y + 0.5 * h)
            l_hauteur = l_max - l_min

            l_pos = img[l_min:l_max, x:x + w]
            lunettes = cv.resize(lunettes_img, (w, l_hauteur))

            for i in range(l_hauteur):
                for j in range(w):
                    if lunettes[i, j, 3] != 0:
                        l_pos[i, j] = lunettes[i, j, :3]

            cv.imshow('Lunettes', img)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
