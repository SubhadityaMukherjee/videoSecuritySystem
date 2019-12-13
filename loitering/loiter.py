import numpy as np
import imutils
import dlib
import cv2
import math
import time

# TO ADD : TIME spent in frame

classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

net = cv2.dnn.readNetFromCaffe("models/model.prototxt",
                               "models/ssd.caffemodel")

video = cv2.VideoCapture("data/walker.mp4")


objectLoc1, objectLoc2, objectTracker, timer = {}, {}, {}, {
}
currentId = 0
while True:
    (grabbed, frame) = video.read()

    if frame is None:
        break
    frame = imutils.resize(frame, width=600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    toDel = []
    for cid in objectTracker.keys():
        trackingQuality = objectTracker[cid].update(frame)
        if trackingQuality < 7:
            toDel.append(cid)

    for a in toDel:
        objectTracker.pop(a, None)
        objectLoc1.pop(a, None)
        objectLoc2.pop(a, None)

    if len(objectTracker) == 0:
        (h, w) = frame.shape[:2]
        createBlob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
        net.setInput(createBlob)
        detected = net.forward()

        for i in np.arange(0, detected.shape[2]):
            conf = detected[0, 0, i, 2]

            if conf > 0.6:
                idx = int(detected[0, 0, i, 1])
                label = classes[idx]


                if label != "person":
                    continue

                box = detected[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                t = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                t.start_track(frame, rect)
                objectTracker[currentId] = t
                objectLoc1[currentId] = [startX, startY, endX, endY]
                currentId += 1

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (255, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    else:
        for cid in objectTracker.keys():
            object = objectTracker[cid].update(frame)
            pos = objectTracker[cid].get_position()
            startX, startY, endX, endY = int(pos.left()), int(pos.top()), int(
                pos.right()), int(pos.bottom())
            objectLoc2[currentId] = [startX, startY, endX, endY]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0),
                          2)



    cv2.imshow("det", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
video.release()
