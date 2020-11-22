import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

bg = None

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        
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

    radius = int(0.7 * maximum_distance)


    circumference = (2 * np.pi * radius)

    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
    cv2.circle(clone, (cX, cY), radius, (0,255,0), 1)

    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    cv2.imshow("Mask", circular_roi)

    (cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        if (cY  > (y + h)) and ((circumference * 0.2) > c.shape[0]) and (c.shape[0] > circumference * 0.05):
            count += 1
    return count

if __name__ == "__main__":
    accumWeight = 0.5
    camera = cv2.VideoCapture(0)
    num_frames = 0
    calibrated = False

    while(True):
        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented], -1, (0, 0, 255))
                fingers = count(thresholded, segmented, clone)
                if fingers ==0:
                    result = "rock"
                elif fingers == 2:
                    result = "scissors"
                elif fingers ==5:
                     result = "paper"

                cv2.putText(clone, result, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
                #cv2.imshow("Thesholded", thresholded)

        num_frames += 1
        cv2.imshow("Video", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()