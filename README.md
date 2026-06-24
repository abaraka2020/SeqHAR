# SeqHAR: Sequential Human Activity Recognition

This repository contains the official implementation of the **SeqHAR** model, a deep learning architecture designed for context-aware sensor-based human activity recognition.
SeqHAR is a deep learning architecture designed to improve activity recognition by explicitly capturing the natural temporal sequence of human activities. By knowing the sequence of preceding activities, the model learns sequential dependencies, enabling more accurate recognition of the current activity using the knowledge of the previous activity. The proposed model leverages both local and temporal dependencies across adjacent windows through two pipelines. The first directly classifies the current activity from the current window, while the second learns the sequence of activities and utilizes the previous window to infer the current activity based on learning the activity sequence from the local and temporal context. A cross-attention mechanism is integrated within these pipelines by aligning current features with both past and present contexts, thereby refining the recognition process. A probability-based fusion strategy combines the outputs of both pipelines.

## Model Architecture
![Model Architecture](/images/SeqHar.png)


## 📦 Project Structure
- `data/`: Place for raw datasets and preprocessing scripts.
- 'src/`: Core implementation of the SeqHAR model.


## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.9+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
