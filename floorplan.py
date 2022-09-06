import cv2
import numpy as np
from skimage import draw

class FloorPlan:
    
    def __init__(self, path='', scale=100):
        self.image_path = path
        self.image = None
        self.floorplan = None
        self.x_size = 0
        self.y_size = 0
        self.scale = scale
        self.accessPoints = []
        if path:
            self.loadFromFile(path, scale)
        
    def rescaleImg(self, scale):
        if scale == 100:
            return
        width = int((self.image.shape[1] * scale) / 100)
        height = int((self.image.shape[0] * scale) / 100)
        dim = (width, height)
        #print(f'Rescaling image with factor of {scale} from {self.image.shape} to {dim}')
        self.image = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)
        self.y_size, self.x_size, _ = self.image.shape

    def loadFromFile(self, image_path, scale=100):
        self.image_path = image_path
        try:
            self.image = cv2.imread(image_path)
            self.y_size, self.x_size, _ = self.image.shape
            self.rescaleImg(scale)
        except:
            raise
    
    def loadFromArray(self, data : np.ndarray, scale=100):
        self.image = data
        self.y_size, self.x_size = self.image.shape
        self.rescaleImg(scale)


    def getWallMap(self, morph=True, kernel = (5, 5)):
        self.removeAccessPoints()
        #Convert to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        #Threshold image to get black and white binary data
        (self.thresh, self.im_bw) = cv2.threshold(self.gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        morph_img = self.im_bw
        #Morphological Transform (remove small details)
        if morph:
            self.kernel = np.ones(kernel, np.uint8)
            self.dilation = cv2.dilate(self.im_bw, self.kernel)
            self.erosion = cv2.erode(self.dilation, self.kernel, iterations=1)
            morph_img = self.erosion
                
        #Use direct logic (Wall = 255, floor = 0)
        self.invert = np.invert(morph_img)
        
        #Convert to binary
        self.binary = self.invert/255
        
        self.wallmap = self.binary
        
        return self.wallmap
    
    def countWalls(self, src_x, src_y, dst_x, dst_y, debug=False):
        #Get line points
        line_r, line_c = draw.line(int(src_y), int(src_x), int(dst_y), int(dst_x))
    
        num_pts = len(line_r)
        walls = 0
        same_wall = False
        if debug: 
            print(f'From [{src_x},{src_y}] to [{dst_x},{dst_y}]')
        for idx in range(num_pts):
            if self.wallmap[line_r[idx],line_c[idx]]:
                #Found a wall
                if not same_wall:
                    #Check if we are not inside the same wall
                    walls += 1
                    if debug: 
                        print(f'Wall: [{line_r[idx]},{line_c[idx]}]')
                    same_wall = True
            elif same_wall == True:
                #Left the wall, can count again
                same_wall = False
        if debug: 
            print(f'Total walls: {walls}')
        return walls
    
    def findAccessPoints(self, color='blue'):
        lower_blue = np.array([101,50,38])
        upper_blue = np.array([130,255,255])
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(self.image, self.image, mask= mask)

        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
            M = cv2.moments(contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            self.accessPoints.append((cx,cy,x,y,w,h))

        return self.accessPoints

    def removeAccessPoints(self):
        if len(self.accessPoints) > 0:
            return

        #Paint AP locations white
        self.findAccessPoints()
        for _,_,x,y,w,h in self.accessPoints:
            cv2.rectangle(self.image, (x,y), (x+w, y+h), (255,255,255), -1) 