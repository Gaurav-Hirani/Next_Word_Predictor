# **🧠 Next Word Prediction (LSTM)**

An end-to-end Natural Language Processing (NLP) project that utilizes **Long Short-Term Memory (LSTM)** Recurrent Neural Networks to predict the most probable next word in a given sequence of text.

## **🚀 Overview**

Predictive text is a core component of modern digital interfaces. This project demonstrates a complete machine learning pipeline:

1. **Data Preprocessing**: Cleaning and tokenizing a dataset of over 3,000 quotes.  
2. **Modeling**: Building a deep learning model using an Embedding layer and LSTM to capture semantic and sequential relationships.  
3. **Deployment**: Creating an interactive web application using **Streamlit** for real-time inference.

## **🛠️ Tech Stack**

* **Language**: Python 3.8+  
* **Deep Learning**: TensorFlow, Keras  
* **Web Framework**: Streamlit  
* **Data Science**: Pandas, NumPy, Scikit-learn  
* **Serialization**: Pickle (for Tokenizer and Metadata)

## **📁 Repository Structure**

| File | Description |
| :---- | :---- |
| app.py | The main Streamlit application providing the user interface and prediction logic. |
| codefile.ipynb | The primary development notebook: includes data cleaning, N-gram generation, and LSTM training. |
| RNNimplementation.ipynb | Experimental notebook exploring the foundational concepts of Recurrent Neural Networks. |
| lstm\_model (1).h5 | The pre-trained weights and architecture of the LSTM model. |
| tokenizer.pkl | The pickled Keras Tokenizer containing the word-to-index mapping (\~2,700+ words). |
| max\_len.pkl | A saved integer representing the maximum sequence length used during training. |
| qoute\_dataset.csv | The source dataset containing quotes used to teach the model linguistic patterns. |

## **🧠 Model Architecture**

The neural network architecture designed in codefile.ipynb consists of:

* **Embedding Layer**: Maps vocabulary words into a dense 100-dimensional vector space.  
* **LSTM Layer**: 150 hidden units that process sequences and manage long-term dependencies.  
* **Dense Layer**: A Softmax-activated output layer that predicts the probability distribution across the entire vocabulary.

## **⚙️ Installation & Usage**

### **1\. Prerequisites**

Ensure you have the following packages installed:

pip install tensorflow streamlit pandas numpy

### **2\. Run the Application**

Navigate to the project directory and execute:

streamlit run app.py

### **3\. Using the App**

1. Enter a starting phrase (e.g., *"The world is"*).  
2. Click **Predict Next Word**.  
3. The model will process the input and display its top prediction based on the training data context.

## **📊 Performance Insights**

* **Dataset Diversity**: By training on professional quotes, the model has learned a wide variety of sentence structures.  
* **Inference Speed**: The application uses @st.cache\_resource to load the model and tokenizer into memory only once, ensuring instantaneous results.  
* **Convergence**: The training process in codefile.ipynb shows steady loss reduction, indicating effective pattern recognition.

## **🔮 Future Enhancements**

* **Beam Search**: Implement beam search to provide the top 3-5 most likely word suggestions.  
* **Transfer Learning**: Fine-tune the model using a larger corpus like the Wikipedia or Gutenberg datasets.  
* **Transformer Migration**: Transition from LSTM to an Attention-based Transformer architecture (like a mini-GPT) for even greater context awareness.

