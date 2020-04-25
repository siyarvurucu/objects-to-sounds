import cv2

videoPath = "http://192.168.1.2:4747/mjpegfeed?640x480"
cap = cv2.VideoCapture(videoPath)

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # selecting codec
out = cv2.VideoWriter('singingobjects.avi',fourcc, 20.0, (640,480))

print("Recording video... Press 'q' to quit")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

