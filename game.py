import cv2
import numpy as np
import math
import dlib
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
class Object:
    def __init__(self, size=50):
        self.logo_org = cv2.imread('zhanat.jpg')
        self.size = size
        self.logo = cv2.resize(self.logo_org, (size, size))
        img2gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        _, logo_mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        self.logo_mask = logo_mask
        self.x = 295
        self.y = 240
        self.score = 0

    def insert_object(self, frame):
        roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
        roi[np.where(self.logo_mask)] = 0
        roi += self.logo

    def update_position(self, x,y):
        return self.x, self.y
    
# Let's create the object
target_for_player1 = Object()
target_for_player2 = Object()

""" Detect collision between a rectangle and circle. """
def collision(rleft, rtop, width, height,   # rectangle definition
              center_x, center_y, radius):  # circle definition

    # complete boundbox of the rectangle
    rright, rbottom = rleft + width, rtop + height

    # bounding box of the circle
    cleft, ctop     = center_x-radius, center_y-radius
    cright, cbottom = center_x+radius, center_y+radius

    # trivial reject if bounding boxes do not intersect
    if rright < cleft or rleft > cright or rbottom < ctop or rtop > cbottom:
        return False  # no collision possible

    # check whether any point of rectangle is inside circle's radius
    for x in (rleft, rleft+width):
        for y in (rtop, rtop+height):
            # compare distance between circle's center point and each point of
            # the rectangle with the circle's radius
            if math.hypot(x-center_x, y-center_y) <= radius:
                return True  # collision detected

    # check if center of circle is inside rectangle
    if rleft <= center_x <= rright and rtop <= center_y <= rbottom:
        return True  # overlaid

    return False  # no collision detected

def landmark(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray,1)
    noses_x = []
    noses_y = []        
            
    for face in range(0,len(faces)):
        faces = hog_face_detector(gray)
        noses_x = []
        noses_y = []      
        face_landmarks = dlib_facelandmark(gray, faces[face])
        nose_x = face_landmarks.part(30).x
        nose_y = face_landmarks.part(30).y
        noses_x.append(nose_x)
        noses_y.append(nose_y) 
        rad = 5
        cv2.circle(frame, (nose_x, nose_y), rad, (0, 255, 255), cv2.FILLED)   

def main():
    print("Hello World!")
    while True:
        _, frame = webcam.read()
        frame = cv2.flip(frame, 1)

        #target_for_player1.insert_object(frame)
        #target_for_player2.insert_object(frame)
        landmark(frame)
        cv2.imshow('frame', frame)
        
        k = cv2.waitKey(1)
        if k == 27: # ASCII code 27 in decimal - ESC
            webcam.release() 
            cv2.destroyAllWindows() 
            break  

if __name__ == "__main__":
    main()