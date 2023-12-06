import cv2 as cv

# Initialize paths and cascade classifier
sunglasses_paths = [
    'C:/Users/pc/Desktop/DetectionOpenCv/source/lunettes/1.png',
    'C:/Users/pc/Desktop/DetectionOpenCv/source/lunettes/2.png',
    'C:/Users/pc/Desktop/DetectionOpenCv/source/lunettes/3.png'
]

sunglasses_index = 0
visage_cascade = cv.CascadeClassifier('C:/Users/pc/Desktop/DetectionOpenCv/cascade/visage.xml')

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv.flip(img, 1)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    visages = visage_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=4)

    for x, y, w, h in visages:
        sunglasses_img = cv.imread(sunglasses_paths[sunglasses_index], -1)
        if w * h > 2000:
            l_min = int(y + 0.3 * h)
            l_max = int(y + 0.5 * h)
            l_hauteur = l_max - l_min

            l_pos = img[l_min:l_max, x:x + w]
            sunglasses = cv.resize(sunglasses_img, (w, l_hauteur))

            for i in range(l_hauteur):
                for j in range(w):
                    if sunglasses[i, j, 3] != 0:
                        l_pos[i, j] = sunglasses[i, j, :3]

    cv.imshow('Sunglasses', img)

    # Check for key presses
    key = cv.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('1'):
        sunglasses_index = 0  # Press '1' to switch to the first sunglasses image
    elif key == ord('2'):
        sunglasses_index = 1  # Press '2' to switch to the second sunglasses image
    elif key == ord('3'):
        sunglasses_index = 2  # Press '3' to switch to the third sunglasses image

cap.release()
cv.destroyAllWindows()
