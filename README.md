# Object detection using a Single Shot Detector

In this project, my goal is to train a neural network to detect both robots and balls in the context of [RoboCup SPL](https://spl.robocup.org/).
I have implemented a Single Shot Detector (SSD) following the work of Liu et al.[^1]. The architecture of the initial network is based on the works of [BHuman](https://github.com/bhuman/BHumanCodeRelease).
However, the implementation of BHuman is limited to detecting only Robots. This project is my attempt to extend the network such that it detects both robots and balls.

The training data for this project was obtained from [RoboEireann](https://roboeireann.maynoothuniversity.ie/research/SPLObjDetectDatasetV2.zip)[^2]. A sample of nine images is provided below.
![](visualization/example.png)
[^1]:[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
[^2]:[Faster YOLO-LITE: Faster Object Detection on Robot and Edge Devices](https://link.springer.com/chapter/10.1007/978-3-030-98682-7_19)

# Setup

To set up the project, follow these steps:

1. Clone the repository to your local machine.

```
git clone https://github.com/jonathan-hellwig/robot-detection.git
```

2. Navigate to the project directory.

```
cd robot-detection
```

3. Create a virtual environment and activate it.

```
python3 -m venv .venv
source .venv/bin/activate
```

4. Install the required packages.

```
pip install -r requirements.txt
```
# Training

The files `robot_detection_train.ipynb` and `synthetic_data_train.ipynb` contain the code to train a JetNet on the RoboEireann data and synthetic data respectively.
