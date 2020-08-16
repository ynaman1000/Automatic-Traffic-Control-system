# python3 main.py -y yolo-coco -i phase1.mp4 -o phase1_final.avi -n 2 -p 1

import argparse
import os
import cv2
import time
import numpy as np
import imutils
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True)
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-n", "--video_no", required=True)
ap.add_argument("-p", "--phase_no", required=False)
args = vars(ap.parse_args())

video_no = int(args["video_no"])
if args["phase_no"]:
	phase_no = int(args["phase_no"])
else:
	phase_no = -1


# Class defining Point
class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def get_x(self):
		return self.x

	def get_y(self):
		return self.y


# Class defining Vector joining two points
class Vector:
	def __init__(self, a, b):
		self.x = b.get_x() - a.get_x()
		self.y = b.get_y() - a.get_y()

	def get_x(self):
		return self.x

	def get_y(self):
		return self.y

	def dot(self, v):
		return self.x*v.get_x() + self.y*v.get_y()


# Function for checking whether the point P is inside polygon ABCD
def is_inside(P, A, B, C, D):
	AP = Vector(A, P)
	AB = Vector(A, B)
	AD = Vector(A, D)
	costheta1 = AP.dot(AB)/(AP.dot(AP))**0.5
	costheta2 = AD.dot(AB)/(AD.dot(AD))**0.5

	BP = Vector(B, P)
	BA = Vector(B, A)
	BC = Vector(B, C)
	costheta3 = BP.dot(BA)/(BP.dot(BP))**0.5
	costheta4 = BC.dot(BA)/(BC.dot(BC))**0.5
	
	DP = Vector(D, P)
	DA = Vector(D, A)
	DC = Vector(D, C)
	costheta5 = DP.dot(DA)/(DP.dot(DP))**0.5
	costheta6 = DC.dot(DA)/(DC.dot(DC))**0.5

	return (costheta1>costheta2) and (costheta3>costheta4) and (costheta5>costheta6)


CONF = 0.5
THRES = 0.3

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# print(LABELS)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(4, 3), dtype="uint8")
LABEL_COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Loading pre-trained YOLOv3 model
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initializing video writer to write the output video
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
# Counting total number of frames in the video
prop = cv2.CAP_PROP_FRAME_COUNT
total = int(vs.get(prop))

# Looping through each frame of the video
break_count = 0
coordinates = np.zeros((1, 2))
V = np.zeros((1, 3))

cnt = -1
while True:
	cnt += 1
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# Converting image into a blob (N,C,H,W)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)		# Output from neural network
	end = time.time()

	boxes = []
	centers = []
	confidences = []
	IDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > CONF:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				centers.append([centerX, centerY])
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				IDs.append(classID)

	# Non-Maxima Suppression
	indexs = cv2.dnn.NMSBoxes(boxes, confidences, CONF, THRES)

	if break_count == 0:
		if video_no == 3: 
			A = Point(165, 0)
			B = Point(205, 0)
			C = Point(237, 0)
			D = Point(279, 0)
			E = Point(451, 200)
			F = Point(325, 200)
			G = Point(229, 200)
			I = Point(140, 200)
		elif video_no == 2:
			A = Point(354, 0)
			B = Point(389, 0)
			C = Point(420, 0)
			D = Point(458, 0)
			E = Point(630, 200)
			F = Point(507, 200)
			G = Point(408, 200)
			I = Point(324, 200)

	ln_wdth = 2

	cv2.line(frame,(A.get_x(),0),(B.get_x(), 0),(0,0,255),ln_wdth)
	cv2.line(frame,(I.get_x(),200),(G.get_x(), 200),(0,0,255),ln_wdth)

	cv2.line(frame,(A.get_x(),0),(I.get_x(), 200),(0,0,255),ln_wdth)
	cv2.line(frame,(B.get_x(),0),(G.get_x(), 200),(0,0,255),ln_wdth)


	cv2.line(frame,(B.get_x(),0),(C.get_x(), 0),(0,0,255),ln_wdth)
	cv2.line(frame,(G.get_x(),200),(F.get_x(), 200),(0,0,255),ln_wdth)

	cv2.line(frame,(B.get_x(),0),(G.get_x(), 200),(0,0,255),ln_wdth)
	cv2.line(frame,(C.get_x(),0),(F.get_x(), 200),(0,0,255),ln_wdth)


	cv2.line(frame,(C.get_x(),0),(D.get_x(), 0),(0,0,255),ln_wdth)
	cv2.line(frame,(F.get_x(),200),(E.get_x(), 200),(0,0,255),ln_wdth)

	cv2.line(frame,(C.get_x(),0),(F.get_x(), 200),(0,0,255),ln_wdth)
	cv2.line(frame,(D.get_x(),0),(E.get_x(), 200),(0,0,255),ln_wdth)

	if phase_no == -1:
		cv2.circle(frame, (60, 30), 25, (0, 255, 0), 2)
		cv2.circle(frame, (60, 90), 25, (0, 255, 255), 2)
		cv2.circle(frame, (60, 150), 25, (0, 0, 255), 2)
	else:
		cv2.circle(frame, (150, 30), 25, (0, 255, 0), 2)
		cv2.circle(frame, (150, 90), 25, (0, 255, 255), 2)
		cv2.circle(frame, (150, 150), 25, (0, 0, 255), 2)


	if phase_no == 1:
		if cnt<32*30:
			cv2.circle(frame, (150, 30), 25, (0, 255, 0), -1)
		elif cnt<35*30:
			cv2.circle(frame, (150, 90), 25, (0, 255, 255), -1)
		else:
			cv2.circle(frame, (150, 150), 25, (0, 0, 255), -1)
	elif phase_no == 2:
		if cnt>=35*30 and cnt<(35+24)*30:
			cv2.circle(frame, (150, 30), 25, (0, 255, 0), -1)
		elif cnt>=(35+24)*30 and cnt<(35+27)*30:
			cv2.circle(frame, (150, 90), 25, (0, 255, 255), -1)
		else:
			cv2.circle(frame, (150, 150), 25, (0, 0, 255), -1)
	elif phase_no == 3:
		if cnt>=(35+27)*30:
			cv2.circle(frame, (150, 30), 25, (0, 255, 0), -1)
		# elif cnt<30*30:
			# cv2.circle(frame, (150, 90), 25, (0, 255, 255), -1)
		else:
			cv2.circle(frame, (150, 150), 25, (0, 0, 255), -1)
	elif phase_no == 4:
		# if cnt<25*30:
		# 	cv2.circle(frame, (150, 30), 25, (0, 255, 0), -1)
		# elif cnt<28*30:
		# 	cv2.circle(frame, (150, 90), 25, (0, 255, 255), -1)
		# else:
		cv2.circle(frame, (150, 150), 25, (0, 0, 255), -1)


	if len(indexs) > 0:
		sum = 0
		pre_coordinates = coordinates
		coordinates = np.zeros((1, 3))
		for i in indexs.flatten():
			if IDs[i] in [1, 2, 3, 5, 7]:
				# Extracting the bounding box coordinates
				(x, y) = (centers[i][0], centers[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				P = Point(x, y)

				if is_inside(P, A, D, E, I):
					if (coordinates[0, 0] == 0) and (coordinates[0, 1] == 0):
						coordinates = np.array([[x, y, 1]])
					else:
						coordinates = np.append(coordinates, [[x, y, 1]], axis=0)
				else:
					continue

				# Setting weights based on the vehicle type (PCU)
				if IDs[i] in [5, 7]:
					coordinates[-1, -1] = 3
				elif IDs[i] in [1, 3]:
					coordinates[-1, -1] = 0.5

				# Choosing different color for different lanes
				if is_inside(P, A, B, G, I):
					color = [int(c) for c in COLORS[0]]
				elif is_inside(P, B, C, F, G):
					color = [int(c) for c in COLORS[1]]
				else:
					color = [int(c) for c in COLORS[2]]

				# Drawing a bounding box rectangle and label on the frame
				cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 1)
				text = "{}: {:.4f}".format(LABELS[IDs[i]], confidences[i])
				cv2.putText(frame, text, (int(x-w/2), int(y-h/2 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
				sum += 1

		# Printing number of vehicles crossing each lane
		color = [0, 0, 0]
		cv2.putText(frame, repr(sum), (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
		cv2.putText(frame, repr(V[0, 0]), (341, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		cv2.putText(frame, repr(V[0, 1]), (428, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		cv2.putText(frame, repr(V[0, 2]), (569, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

	coordinates = coordinates[coordinates[:,1].argsort()]

	i=0

	while cnt>0:
		if (pre_coordinates[-1-i, 1]-coordinates[-1, 1] > 5):
			# print(coordinates)
			# print(pre_coordinates)
			P = Point(pre_coordinates[-1-i, 0], pre_coordinates[-1-i, 1])
			if is_inside(P, A, B, G, I):
				V[0, 0] += pre_coordinates[-1-i, -1]
				print(i, 1, pre_coordinates[-1-i, -1])
			elif is_inside(P, B, C, F, G):
				V[0, 1] += pre_coordinates[-1-i, -1]
				print(i, 2, pre_coordinates[-1-i, -1])
			else:
				V[0, 2] += pre_coordinates[-1-i, -1]
				print(i, 3,pre_coordinates[-1-i, -1])
		else:
			break
		i += 1

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
		if total > 0:
			elap = (end - start)
			print(total)
			print("Estimated total time to finish: {:.4f}".format(elap * total))

	writer.write(frame)
	# cv2.imshow("Frame", frame)
	# cv2.waitKey(0)
	if break_count == -1:
		break
	else:
		break_count += 1

print(V)

writer.release()
vs.release()