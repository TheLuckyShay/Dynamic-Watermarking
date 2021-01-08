import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image

######################################################################################
#Constants
######################################################################################
MAX_LINE_GAP = 100
WATERMARK = cv2.imread("WatermarkText.png", cv2.IMREAD_UNCHANGED)
IMAGE_NAME = '2802.jpeg'
######################################################################################
#Functions
######################################################################################

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
  dim = None
  (h, w) = image.shape[:2]

  if width is None and height is None:
      return image
  if width is None:
      r = height / float(h)
      dim = (int(w * r), height)
  else:
      r = width / float(w)
      dim = (width, int(h * r))

  return cv2.resize(image, dim, interpolation=inter)

def rotateImage(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def cvToPil(cvImage):
  
  # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that 
  # the color is converted from BGR to RGB
  color_coverted = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
  pilImage = Image.fromarray(color_coverted)
  return pilImage

def pilToCv(pilImage):
  # use numpy to convert the pil_image into a numpy array
  npImage=np.array(pilImage)  

  # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that 
  # the color is converted from RGB to BGR format
  cvImage=cv2.cvtColor(npImage, cv2.COLOR_RGB2BGR) 
  return cvImage

def pasteWatermark(image,angle,tempx,tempy):
  tiltedWatermark = rotateImage(WATERMARK, angle)
  h1, _ = WATERMARK.shape[:2]
  h2, _ = tiltedWatermark.shape[:2]
  pilBackground = cvToPil(image)
  pilWatermark = cvToPil(tiltedWatermark)
  mask_im = pilWatermark.convert('L')
  print(x1)
  print(y1)
  print(tempx)
  print(tempy)
  radAngle = angle * np.pi/180
  print(h1*math.sin(radAngle))
  if angle > 0:
    tempx -= int(h1*math.sin(radAngle))
    tempy -= h2
  elif angle < 0:
    tempy -= int(h1*math.cos(radAngle))
  else:
    tempy -= h1

  pilBackground.paste(pilWatermark,(tempx,tempy),mask_im)
  done = pilToCv(pilBackground)
  return done

######################################################
#Main
######################################################
original = cv2.imread(IMAGE_NAME,cv2.IMREAD_UNCHANGED)
img = cv2.imread(IMAGE_NAME,0)
edges = cv2.Canny(img,300,300)

i = 300
lines = cv2.HoughLinesP(edges,rho = 0.5,theta = 1*np.pi/180,threshold = 100,minLineLength = 10,maxLineGap = MAX_LINE_GAP)

while lines.size > 160:
  i += 10
  edges = cv2.Canny(img,i,i)
  lines = cv2.HoughLinesP(edges,rho = 0.5,theta = 1*np.pi/180,threshold = 100,minLineLength = 10,maxLineGap = MAX_LINE_GAP)
while lines.size < 160:
  i -= 10
  edges = cv2.Canny(img,i,i)
  lines = cv2.HoughLinesP(edges,rho = 0.5,theta = 1*np.pi/180,threshold = 100,minLineLength = 10,maxLineGap = MAX_LINE_GAP)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

angleVar = 20

goodLines = []
lineDone= original

while True:
  for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
    if angle >= (-1 * angleVar) and angle <= angleVar:
#
# 
      cv2.line(lineDone, (x1, y1), (x2, y2), (0, 0, 0), 4)
      goodLines.append(line)
#      lineDone = pasteWatermark(lineDone, angle, x1, y1)
  if len(goodLines) == 0:
    angleVar += 2
  else: break

goodLines2 = []

#while len(goodLines) > 3:
imageWidth, _ = original.shape[:2]
for line in goodLines:
  x1, y1, x2, y2 = line[0]
  lineLengthX = x2-x1
  if lineLengthX >= (imageWidth/9) and lineLengthX <= (imageWidth/3):
    goodLines2.append(line)
    cv2.line(lineDone, (x1, y1), (x2, y2), (0, 0, 255), 4)
goodLines = []




#remove ones that overlap, are too long, or are too close together, keeping in mind that count must
##not go less than 3
#prioritize the ones closer to the middle
 #sort them and assign priority by how close to the middle they are, perhaps using priority list
#if any but one are left, filter by the color above the line (should be dark so the white watermark
##can show)
  #if the color is light, there can alternatively be a black watermark
#comment: in the end, convert the final result to cv to be able to show it


######################
#Testing
######################

lineDone = ResizeWithAspectRatio(lineDone, height=900)
cv2.imshow("lines", lineDone)
plt.show()



