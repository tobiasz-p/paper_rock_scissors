import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

bg = None

def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, accumWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    countours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(countours) == 0:
        return
    else:
        segmented = max(countours, key=cv2.contourArea)
        return (thresholded, segmented)


def count(thresholded, segmented, clone):
    chull = cv2.convexHull(segmented)

    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    radius = int(0.85 * maximum_distance)
    l = (2 * np.pi * radius)

    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
    cv2.circle(clone, (cX, cY), radius, (0,255,0), 1)

    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    cv2.imshow("Mask", circular_roi)

    (countours, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    for countour in countours:
        (_, y, _, h) = cv2.boundingRect(countour)
        if (1.25 * cY  > (y + h)) and (countour.shape[0] < (0.25 * l)):
            count += 1

    return count

if __name__ == "__main__":
    accumWeight = 0.5
    path = "./videos/video-4.mkv"
    camera = cv2.VideoCapture(path)
    num_frames = 0
    calibrated = False

    while(True):
        (_, frame) = camera.read()

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, accumWeight)
        else:
            hand = segment(gray)
            if hand is not None:
                #result =''
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented], -1, (0, 0, 255))
                fingers = count(thresholded, segmented, clone)
                if fingers ==0:
                    result = "rock"
                elif fingers == 2:
                    result = "scissors"
                elif fingers ==5:
                     result = "paper"
                     
                cv2.putText(clone, result, (70, 45), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2)
                cv2.imshow("Thesholded", thresholded)

        num_frames += 1
        cv2.imshow("Video", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()