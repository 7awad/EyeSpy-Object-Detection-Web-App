# EyeSpy Object Detection Web App


## Overview
EyeSpy Object Detection is a web-based application that allows users to perform object detection on uploaded images using a pre-trained YOLOv3 model. This project uses **Flask** for the web interface, **OpenCV** for object detection, and the YOLOv3 model to recognize and label objects in images. The user can easily upload an image, and the system will return the same image with detected objects highlighted.
![image](https://github.com/user-attachments/assets/b4c09456-17d8-4b44-976a-5488384cdaed)





## Technologies Used
- **Flask**: Python micro web framework to create the web interface.
- **OpenCV**: Library used for image processing and object detection.
- **YOLOv3 (You Only Look Once)**: Pre-trained model used for real-time object detection.
- **HTML/CSS**: To build a simple and elegant front-end user interface.

## Project Directory Structure
```
EyeSpy Real-Time Object Detection/
    ├── .venv/
    ├── static/
    │       ├── uploaded/ (to store uploaded images)
    │       └── css/
    │             └── styles.css (CSS for styling the web page)
    ├── templates/
    │       └── index.html (HTML for the web page)
    ├── yolo/
    │       ├── coco.names
    │       ├── yolov3.cfg
    │       └── yolov3.weights
    └── app.py (Flask server script)
```

## How to Use
### Prerequisites
- **Python 3.x**: Make sure Python is installed on your system.
- **Pip**: Ensure you have pip installed to manage Python libraries.
- **Virtual Environment (Optional)**: Recommended for dependency management.

### Setup Instructions
1. **Clone the Repository**
   ```sh
   git clone https://github.com/7awad/EyeSpy-Object-Detection-Web-App
   cd EyeSpy Real-Time Object Detection
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```sh
   python -m venv .venv
   source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
   ```

3. **Install the Required Libraries**
   ```sh
   pip install Flask opencv-python opencv-python-headless numpy
   ```

4. **Download YOLOv3 Weights**
   - The weights file (`yolov3.weights`) is too large to be included in the repository. Please download it from [this link](https://www.kaggle.com/datasets/shivam316/yolov3-weights?resource=download).
   - Unzip the downloaded file and place `yolov3.weights` in the `yolo/` folder of your project.

5. **Run the Application**
   ```sh
   python app.py
   ```
   The application will start running on `http://127.0.0.1:5000/`.

6. **Using the Web Interface**
   - Go to `http://127.0.0.1:5000/` in your browser.
   - Upload an image for object detection.
   - View the processed image with detected objects highlighted.

## Libraries Used
- **Flask**: To create the web server.
- **OpenCV (cv2)**: For reading, processing, and modifying images.
- **NumPy**: To perform numerical operations.

## How It Works
1. **Frontend (HTML/CSS)**: Users can access a web page where they can upload an image.
2. **Backend (Flask)**: Flask handles the image upload, saves it in the `static/uploaded/` folder, and processes it with the YOLOv3 model.
3. **Object Detection (YOLO + OpenCV)**:
   - The uploaded image is read using OpenCV.
   - YOLOv3 is used to detect objects, and the bounding boxes are drawn around detected items.
4. **Result**: The image is saved and displayed with detected objects.

## Notes
- Make sure to download the YOLOv3 weights file before running the application.
- This project uses YOLOv3, which is trained on the COCO dataset, capable of detecting **80 different classes** of objects (e.g., people, cars, animals).

## License
This project is open source and available under the MIT License.

Feel free to contribute or improve upon the project!

