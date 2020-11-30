import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import time

background = None

def avg(image, w):
    global background
    if background is None:
        background = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, background, w)


def segmentation(image, th=25):
    global background
    diff = cv2.absdiff(background.astype("uint8"), image)
    thresholded = cv2.threshold(diff, th, 255, cv2.THRESH_BINARY)[1]
    countours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(countours) == 0:
        return
    else:
        segmented = max(countours, key=cv2.contourArea)
        return (thresholded, segmented)

def count(thresholded, segmented, frame_cp):
    hull = cv2.convexHull(segmented)
    top    = tuple(hull[hull[:, :, 1].argmin()][0])
    bottom = tuple(hull[hull[:, :, 1].argmax()][0])
    left   = tuple(hull[hull[:, :, 0].argmin()][0])
    right  = tuple(hull[hull[:, :, 0].argmax()][0])
    x = int((left[0] + right[0]) / 2)
    y = int((top[1] + bottom[1]) / 2)
    dist = pairwise.euclidean_distances([(x, y)], Y=[left, right, top, bottom])[0]
    max_dist = dist[dist.argmax()]
    r = int(0.8 * max_dist)
    l = (2 * np.pi * r)
    roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(roi, (x, y), r, 255, 1)
    cv2.circle(frame_cp, (x, y), r, (0,255,0), 1)
    roi = cv2.bitwise_and(thresholded, thresholded, mask=roi)
    (countours, _) = cv2.findContours(roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    for countour in countours:
        _, y1, _, h = cv2.boundingRect(countour)
        if ((y + (y * 0.25)) > (y1 + h)) and ((l * 0.25) > countour.shape[0]):
            count += 1
    return count

if __name__ == "__main__":
    w = 0.5
    path = "./videos/1.mkv"
    input = cv2.VideoCapture(path)
    frames_counter = 0
    result = None

    while(True):
        _, frame = input.read()
        frame = cv2.flip(frame, 1)
        frame_cp = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if frames_counter < 30:
            avg(gray, w)
        else:
            hand = segmentation(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(frame_cp, [segmented], -1, (0, 0, 255))
                fingers = count(thresholded, segmented, frame_cp)
                if fingers == 0:
                    result = "rock"
                elif fingers == 2:
                    result = "scissors"
                elif fingers == 5:
                     result = "paper"
                if result is not None:
                    cv2.putText(frame_cp, result, (70, 45), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2)
                cv2.imshow("Thesholded", thresholded)
                time.sleep(0.01)

        frames_counter += 1
        cv2.imshow("Video", frame_cp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

input.release()
cv2.destroyAllWindows()