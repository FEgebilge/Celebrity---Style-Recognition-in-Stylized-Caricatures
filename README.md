# Celebrity & Style Recognition in Stylized Caricatures

This repository contains the project files for the course project of COMP 4437 Artificial Neural Networks, Spring 2024. The project aims to build a deep neural network that recognizes the celebrity and the style of caricature images. The dataset includes caricatures of 20 subjects in 6 styles, and the project involves designing and training a multi-label classification network.

## Project Description

Caricature images of celebrities often appear on the internet. The recent advances in AI have made rendering caricatures in various styles easier and even more common than in the past. The goal of this project is to build a deep neural network that recognizes the celebrity and the style of the caricature image. 

The project utilizes a dataset composed of:
- **Training set**: 12 subjects in 6 styles (2160 images)
- **Validation set**: 6 subjects in 6 styles (240 images)
- **Test set**: 6 subjects in 6 styles (240 images)

## Method

The project involves building a neural network model that jointly recognizes the identity and the style by a predictor head for each task. The key components include:
- Multi-label loss function combining identity and style prediction losses.
- Architecture design using a backbone model connected to two prediction heads.
- Optimization and regularization techniques such as transfer learning, learning rate schedulers, and hyperparameter tuning.

## Architecture and Loss Function

The network is designed with a backbone model for feature extraction and two separate heads for identity and style recognition. The loss function is a combination of:
- Cross-Entropy Loss for identity recognition
- Categorical Cross-Entropy Loss for style recognition

## Dataset and Preprocessing

The dataset provided includes caricatures in different styles and is partitioned into training, validation, and test sets. Data augmentation techniques such as random rotations and flips are applied to improve generalization.

## Training and Evaluation

The model is trained using PyTorch with the following steps:
1. Load and preprocess the dataset.
2. Design and compile the neural network.
3. Train the model with appropriate loss functions and optimizers.
4. Evaluate the model on the validation set using accuracy metrics for both identity and style recognition.
5. Fine-tune the model based on validation performance and hyperparameter tuning.

## Results

The performance of the model is evaluated based on:
- Validation set accuracy for identity and style recognition.
    - 90% accuracy for identitiy recognition.
    - 87.5% accurasy for style recognition.
- Test set performance (submitted separately).

## Files

- `COMP4437_2023_24Spring_CourseProject.pdf`: Course project description and requirements.
- `ANN_Celebrity_Recognition.ipynb`: Jupyter notebook containing the implementation details.
- `helper.py`: Helper functions for data loading, preprocessing, and evaluation.
- `ANN_Celebrity_Recognition.pdf`: Detailed project report.
- `example folder` : Example images from test and train datasets.

## Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/FEgebilge/Celebrity---Style-Recognition-in-Stylized-Caricatures.git
   cd celebrity-style-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to train and evaluate the model:
   ```bash
   jupyter notebook ANN_Celebrity_Recognition.ipynb
   ```

## Contributors

- [Fahrettin Ege Bilge](https://github.com/FEgebilge)
- Dr. Arman Savran (Lecturer)
- Onur Kılınç (Assistant)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

