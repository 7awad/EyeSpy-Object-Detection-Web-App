import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Paths to YOLO files
model_config = "yolo/yolov3.cfg"
model_weights = "yolo/yolov3.weights"
model_classes = "yolo/coco.names"

# Load YOLO network
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# Load the COCO class labels
with open(model_classes, "r") as f:
    classes = f.read().strip().split("\n")

# Get the output layer names from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Route to the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle the upload and object detection
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load the uploaded image
        image = cv2.imread(filepath)
        height, width, _ = image.shape

        # Prepare the frame for YOLO (create a blob)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Perform forward pass to get the output
        detections = net.forward(output_layers)

        # Initialize lists to hold detection details
        boxes = []
        confidences = []
        class_ids = []

        # Loop through each of the detections
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Confidence threshold to filter weak detections
                if confidence > 0.5:
                    # Scale the bounding box back to the size of the image
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, box_width, box_height) = box.astype("int")

                    # Calculate top-left corner of the bounding box
                    startX = int(centerX - (box_width / 2))
                    startY = int(centerY - (box_height / 2))

                    # Append to the lists
                    boxes.append([startX, startY, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on the image
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the detected image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + file.filename)
        cv2.imwrite(output_path, image)

        return redirect(url_for('uploaded_file', filename='detected_' + file.filename))

# Route to display the detected image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('index.html', filename=filename)

# Main function to run the Flask server
if __name__ == "__main__":
    app.run(debug=True)