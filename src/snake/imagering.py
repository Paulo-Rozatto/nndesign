import cv2

def fadeIn(img1, img2, steps=2): #pass images here to fade between
    transition = []
    fl = float(steps)

    for IN in range(0,steps):
        fadein = IN/fl
        img = cv2.addWeighted(img1, 1-fadein, img2, 1, 0)
        transition.append(img)
    return transition

def trasitionExtend(frames):
    extended = []
    for i in range(1, len(frames)):
        extended.extend(fadeIn(frames[i - 1], frames[i]))
    return extended