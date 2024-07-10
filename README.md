LinkedIn Game Solver
====================

This project is designed to practice my Computer Vision (CV) skills. It solves a fascinating LinkedIn mini-game called Queen's Game using a backtracking algorithm.

Table of Contents
-----------------

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [License](#license)

Introduction
------------

LinkedIn Game Solver is a project that leverages OpenCV and MediaPipe for hand tracking and solving a Queen's Game using a backtracking algorithm. The game involves placing 'kings' on a board where each cell has a specific color, and no two kings can be on the same row, column, or adjacent cells of the same color.

Features
--------

- Hand tracking using OpenCV and MediaPipe.
- Screen capture to analyze the game board.
- Color detection and classification.
- Backtracking algorithm to solve the Queen's Game.
- Visual representation of the solution on the game board.

Installation
------------

To run this project, you need to have Python installed along with the following libraries:

- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- mss

You can install the required libraries using pip:

\`\`\`bash
pip install opencv-python mediapipe numpy scikit-learn mss
\`\`\`

Usage
-----

To use this project, follow these steps:

1. Clone the repository:
   \`\`\`bash
   git clone https://github.com/your-username/linkedin-game-solver.git
   \`\`\`

2. Navigate to the project directory:
   \`\`\`bash
   cd linkedin-game-solver
   \`\`\`

3. Run the hand tracking script:
   \`\`\`bash
   python HandTack.py
   \`\`\`

4. Run the LinkedIn Queen's Game solver script:
   \`\`\`bash
   python linkedin_queens.py
   \`\`\`

The \`HandTack.py\` script will use your webcam to track hand movements, and the \`linkedin_queens.py\` script will capture the game board from your screen, detect colors, and solve the game using a backtracking algorithm.

Code Structure
--------------

- \`HandTack.py\`: Contains the code for hand tracking using OpenCV and MediaPipe.
- \`linkedin_queens.py\`: Contains the code for capturing the game board, detecting colors, and solving the Queen's Game.

### HandTack.py

This script is responsible for:
- Initializing the hand detector.
- Capturing video from the webcam.
- Tracking hand landmarks.
- Detecting fingers up and calculating distances.

### linkedin_queens.py

This script is responsible for:
- Capturing the game board using screen capture.
- Detecting the dominant color of each cell.
- Matching detected colors to predefined color categories.
- Solving the Queen's Game using a backtracking algorithm.
- Displaying the solved game board with visual markers.

Contributing
------------

Contributions are welcome! If you have any improvements or suggestions, please open an issue or create a pull request.

License
-------

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
