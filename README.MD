# **KoraGeek - Soccer Player Detection and Tracking Project**

[Watch the video](./video/output.mp4)

## **Table of Contents**

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Dataset](#dataset)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## **Introduction**

**Kora Geek** is an innovative football (soccer) tracking system designed to enhance match analysis by leveraging computer vision and machine learning techniques. Built for real-time detection and tracking, this project focuses on analyzing players, the ball, and referees, offering an in-depth view of the game that can be further developed into a powerful analysis tool for coaches, analysts, and enthusiasts.

Kora Geek is built for football enthusiasts, analysts, and developers interested in pushing the boundaries of match analysis through data and visualization. As the project evolves, the potential applications range from real-time tactical analysis to game simulation and player performance reviews.

## **Features**

- **Player Detection**: Detect soccer players in the video frames using object detection techniques.
- **Ball Detection**: Detect the soccer ball in the video frames using object detection techniques.
- **Player Tracking**: Track the movements of players and the ball over time in the video.
- **Team Assignment**: Assign players to teams based on their color or jersey.
- **Speed Estimation**: Estimate the speed of players based on their movement in the field.
- **Top-Down View Projection**: Transform the view of the field to a top-down view using homography matrix and perspective transform.
- **Annotators**: Annotate the players, ball, and other objects in the video frames.
- **Video Processing**: Process the video frame by frame and annotate the frames using the above components.

## **Requirements**

The following packages and tools are required to run the project:

- **Python 3.8+**
- **OpenCV**: For video processing and player detection.
- **NumPy**: For numerical calculations.
- **Supervision**: For visualization and annotations.
- **scikit-learn**: For machine learning algorithms.
- **PyTorch**: For deep learning models.
- **ultralytics**: For YOLOv11 object detection models.
- **roboflow**: For dataset management and annotation.

### **Install dependencies using:**

```bash
pip install -r requirments.txt
```

## **Setup Instructions**

1. Clone the repository:

    ```bash
    git clone https://github.com/mohamed22311/KoraGeek.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained YOLOv11 models for object detection and keypoints detection:
    - [Object Detection Model](https://github.com/mohamed22311/KoraGeek/tree/main/models/weights)
    - [Keypoints Detection Model](https://github.com/mohamed22311/KoraGeek/tree/main/models/weights)

4. Update the paths to the pre-trained models in the code:

   ```python
    object_tracker = ObjectTracker('models\weights\object-detection.pt')
    keypoints_tracker = KeypointTracker('models\weights\keypoints-detection.pt')
    ```

5. Run the main.py file to process the video frames and generate the annotated video.

---

## **Dataset**

The original data used in Kora Geek comes from the **DFL - Bundesliga Data Shootout** Kaggle competition. This data has been further processed to create new datasets, which are hosted and available for download from the **Roboflow Universe**. These datasets enable various use cases within the project, including player detection, ball tracking, and pitch keypoint identification.

| **Use Case**                | **Dataset**                                   |
|-----------------------------|-----------------------------------------------|
| Soccer player detection      | [Download Dataset](https://universe.roboflow.com/bundesligavideos/football-detection-model)                         |
| Soccer pitch keypoint detection | [Download Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi/dataset/15)                      |

The **Roboflow Soccer Player Detection** dataset is a specialized dataset designed to detect and track soccer players during a match. It is built from a collection of 1280x1280 images that have been labeled to identify individual players on the field, with bounding boxes marking their locations. This dataset is ideal for use in computer vision tasks like player tracking, team formation analysis, and movement prediction.

### Key Features

- **Annotated Player Data:** Each image in the dataset contains labeled bounding boxes around the players, making it easy for models to learn how to detect them in real-time.
- **Diverse Perspectives:** The dataset includes images captured from different angles and distances, providing a robust foundation for detecting players in various conditions, such as broadcast views, sideline angles, and overhead shots.
- **High-Quality Annotations:** Precise and detailed annotations ensure that the detection of players is accurate, facilitating further analysis such as player movement tracking and team division.

#### Use Cases

- **Real-Time Player Tracking:** Use this dataset to train models that can detect and track players throughout the match.
- **Tactical Analysis:** Analyze player positioning, formations, and movement patterns using the detected player locations.
- **Performance Insights:** Combine player detection with advanced metrics like speed, distance covered, and heatmaps to gain insights into individual and team performances.

#### Classes

- *Ball*
- *Goalkeeper*
- *Player*
- *Referee*

This dataset is part of the broader **Kora Geek** project and can be used to power soccer analysis tools, providing a solid foundation for detecting players in any football match context.

## **Usage**

1. Run the main.py file to process the video frames and generate the annotated video.
2. The annotated video will be saved in the `output_videos` folder.
3. The annotated video will contain the following annotations:
    - Player bounding boxes
    - Player keypoints
    - Team assignments
    - Ball player assignment
    - Speed estimation
    - Top-down view projection
4. The annotated video can be used for sports analytics and video analysis in soccer matches.

---

### **Running the Project**

Once all the dependencies are installed, run the script with the following command:

```bash
python main.py --input_video soccer_match.mp4
```

This will process the input video file, track the players, and estimate their speeds.

## **Code Overview**

The code consists of the following components:

1. **Tracker**:
    - Object Tracker: Detects players and ball in the frame
    - Keypoints Tracker: Detects keypoints of players in the frame
2. **Team Assigner**: Assigns players to teams based on their color
3. **View Transformer**: Transforms the view of the field to top-down view using homography matrix and perspective transform
4. **Ball Player Assigner**: Assigns ball to the player based on the distance between them and the ball
5. **Speed Estimator**: Estimates the speed of the players based on their movement in the field
6. **Annotators**:
    - Object Annotator: Annotates the players and ball in the frame
    - Keypoints Annotator: Annotates the keypoints of players in the frame
    - Projection Annotator: Annotates the players and ball in the top-down view of the field
7. **Video Processor**: Processes the video frame by frame and annotates the frames using the above components

---

## **Limitations**

- The project may not work well in extreme lighting conditions or occlusions.
- The object detection and keypoints detection models may not generalize well to all soccer match videos.
- The speed estimation may not be accurate in all cases due to variations in player movements and camera angles.

## **Future Work**

- Improve the object detection and keypoints detection models for better accuracy and performance.
- Implement player tracking algorithms for smoother and more accurate tracking.
- Enhance the team assignment and ball player assignment algorithms for better results.
- Add more features such as player identification, player statistics, and game analysis.
- Extend the project to other sports such as basketball, football, and tennis.

---

## **Contributing**

Contributions are welcome! If you have suggestions for new features or improvements, feel free to open a pull request or issue. Please adhere to the project's coding standards and guidelines.

## **License**

This project is licensed under the MIT License. See the LICENSE file for more details.

## **Contact**

For further information, feel free to contact:

- **Mohamed Sameh Rezq**

- Email: <mohamed.sameh.10104@gmail.com>
- LinkedIn: [Mohamed's LinkedIn](https://www.linkedin.com/in/mohamed-rezq-b3036b239/)
