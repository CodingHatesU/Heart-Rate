
# Heart Rate Monitoring System with GUI

This is a Python application that uses OpenCV to monitor a person's heart rate using a webcam or other camera. The application is designed to run on Windows, macOS, and Linux.


## Installation

Clone this repo, or download it.
Then run the following command

For GUI run
```bash
  pip install numpy opencv-python tk pillow sv_ttk matplotlib
```

For Non-GUI
```bash
   pip install numpy opencv-python matplotlib
```
    
## Usage/Examples

* Connect your camera to your computer.
* Run the command ```python gui.py``` to start the GUI application. Or ```python heartrate.py``` to run the Non-GUI application.
* Position your face in front of the camera so that your forehead is visible.
* The application will automatically detect your forehead and track your heart rate in real time.
* The heart rate will be displayed on the screen or console.
## Troubleshooting

If you experience any issues running the application, try the following:
* Ensure you have python installed on your system.
* Ensure that your camera is connected to your computer and functioning properly.
* Ensure that you have installed all necessary dependencies by running the command ```pip install numpy opencv-python tk pillow sv_ttk matplotlib``` or ```pip install numpy opencv-python matplotlib```.
* Try restarting the application or your computer.


## License

[MIT](https://choosealicense.com/licenses/mit/)