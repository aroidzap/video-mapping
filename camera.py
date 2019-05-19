import cv2

class Camera():
    def __init__(self):
        self.cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920), self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.shape = (int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)))

    def capture(self):
        _, frame = self.cam.read()
        return frame
        
    def __del__(self):
        self.cam.release()