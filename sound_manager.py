import cv2
import mediapipe as mp
import time
import math
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence = 0.5)
mpDraw = mp.solutions.drawing_utils 

def findHands(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    # print(self.results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # if draw:
                mpDraw.draw_landmarks(frame, handLms, mphands.HAND_CONNECTIONS)
    return frame

def findPosition(frame, handNo,draw = True):
    lmlist = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            # print(id, cx, cy)
            lmlist.append([id, cx, cy])
            if draw == True:
                cv2.circle(frame, (cx,cy), 10, (255,0,255), cv2.FILLED)
        
    return lmlist

cap = cv2.VideoCapture(0)
pTime = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
# print(minVol,maxVol)

# volbar = 400

while True:
    ret, frame = cap.read()
    frame2 = cv2.flip(frame, 1)
    frame2 = findHands(frame2)
    lmlist = findPosition(frame2,0,False)
    

    if len(lmlist) != 0:
        # print(lmli0st[4], lmlist[8])

        x1,y1 = lmlist[4][1], lmlist[4][2]
        x2,y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(frame2, (x1,y1), 10, (255,0,255), cv2.FILLED)
        cv2.circle(frame2, (x2,y2), 10, (255,0,255), cv2.FILLED)
        cv2.circle(frame2, (cx,cy), 10, (0,255,0), cv2.FILLED)
        cv2.line(frame2, (x1,y1), (x2,y2), (0,255,0), 3)
        length = math.hypot(x2-x1,y2-y1)
        # print(length)

        vol = np.interp(length, [50,300], [minVol, maxVol])
        volBar = np.interp(length, [50,300], [400, 150])
        volPer = np.interp(length, [50,300], [0, 100])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)


        if length<50:
             cv2.circle(frame2, (cx,cy), 10, (255,0,155), cv2.FILLED)

        cv2.rectangle(frame2,(50,150), (85,400),(0,255,0),3)
        cv2.rectangle(frame2,(50,int(volBar)), (85,400),(0,255,0), cv2.FILLED)
        cv2.putText(frame2, f'{int(volPer)}%', (40,450), cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(frame2, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)
    
    cv2.imshow("hand Detection", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()
