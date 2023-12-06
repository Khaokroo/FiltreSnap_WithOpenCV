import cv2 as cv

# Initialize paths and cascade classifiers
sunglasses_paths = [
    'C:/Users/pc/Desktop/DetectionOpenCv/Source/lunettes/1.png',
    'C:/Users/pc/Desktop/DetectionOpenCv/Source/lunettes/2.png',
    'C:/Users/pc/Desktop/DetectionOpenCv/source/lunettes/3.png'
]

hat_path = 'C:/Users/pc/Desktop/DetectionOpenCv/Source/chapeau/1.png'
mustache_path = 'C:/Users/pc/Desktop/DetectionOpenCv/Source/moustache/moustache.png'

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
        # Add sunglasses
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

        # Add hat
        hat_img = cv.imread(hat_path, -1)
        h_min = int(y - 0.2 * h)
        h_max = int(y + 0.1 * h)
        h_hauteur = h_max - h_min

        h_pos = img[h_min:h_max, x:x + w]
        hat = cv.resize(hat_img, (w, h_hauteur))

        for i in range(h_hauteur):
            for j in range(w):
                if hat[i, j, 3] != 0:
                    h_pos[i, j] = hat[i, j, :3]

        # Add mustache
        mustache_img = cv.imread(mustache_path, -1)
        m_min = int(y + 0.7 * h)
        m_max = int(y + 0.9 * h)
        m_hauteur = m_max - m_min

        m_pos = img[m_min:m_max, x:x + w]
        mustache = cv.resize(mustache_img, (w, m_hauteur))

        for i in range(m_hauteur):
            for j in range(w):
                if mustache[i, j, 3] != 0:
                    m_pos[i, j] = mustache[i, j, :3]

    cv.imshow('Face Augmentation', img)

    # Check for key presses
    key = cv.waitKey(0)
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('1'):
        sunglasses_index = 0  # Press '1' to switch to the first sunglasses image
    elif key == ord('2'):
        sunglasses_index = 1  # Press '2' to switch to the second sunglasses image

cap.release()
cv.destroyAllWindows()