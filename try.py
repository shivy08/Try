import numpy as np
import cv2

import tensorflow.keras

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('directions.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    resized=cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA)
    resized=np.fliplr(resized)
    #print(type(frame))
    # Our operations on the frame come here
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)

    normalized_image_array = (resized.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)

    dict = {0:"Up",1:"Down",2:"Left",3:"Right"}
    prediction = prediction[0]
    value = max(prediction)
    label = list(prediction).index(value)
    if value>=0.9900: print(dict[label],value)

    if cv2.waitKey(1) and cv2.getWindowProperty("frame",0)==-1:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()