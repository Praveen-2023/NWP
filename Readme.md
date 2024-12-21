# Next-Word Prediction using MLP

This project uses a Multi-Layer Perceptron (MLP) model to predict the next word in a given text input.

## Setup

1. Clone the repository.
2. Create and activate the virtual environment:

    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On macOS/Linux
    myenv\Scripts\activate  # On Windows
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Place your dataset (`trainingfile.txt`) in the root directory.

## Running the Application

1. To train the model, run:

    ```bash
    python app.py
    ```

2. To run the Streamlit app, use:

    ```bash
    streamlit run app.py
    ```

3. Access the app in your browser at `http://localhost:8501`.
