
# JambaRetriever: A SageMaker-based Model Fine-Tuning and Inference Solution

This repository contains a SageMaker-based solution for fine-tuning the Jamba model for embedding generation and similarity search tasks. The primary components include a Jupyter notebook (`JambaRetriever.ipynb`), along with a training script (`train.py`), an inference script (`inference.py`), and a `requirements.txt` file detailing the necessary dependencies.

The Jupyter notebook is designed to guide you through the entire process, from setting up the environment to running distributed training on SageMaker and deploying the fine-tuned model for inference.

## Components

### 1. Jupyter Notebook: `JambaRetriever.ipynb`
This notebook is the main entry point for users. It contains step-by-step instructions to:
- Set up the necessary SageMaker environment.
- Configure the model, dataset, and training parameters.
- Run distributed training using SageMaker's HuggingFace Estimator.
- Deploy the fine-tuned model for inference.
- Test the deployed model by sending requests and retrieving embeddings.

### 2. Training Script: `train.py`
This Python script handles the model fine-tuning. It uses the Hugging Face library to:
- Load the pre-trained Jamba model and tokenizer.
- Process the input data into suitable formats for training.
- Train the model on a specified dataset.
- Save the fine-tuned model to an Amazon S3 bucket for future use.
- Log metrics and outputs for tracking performance.

### 3. Inference Script: `inference.py`
The `inference.py` script is used for deploying the model and handling inference tasks. Once the model is trained and deployed as an endpoint, this script:
- Loads the fine-tuned model and tokenizer.
- Pre-processes the input text into token embeddings.
- Generates output embeddings based on the input sentences.
- Sends these embeddings for downstream tasks such as similarity searches.

### 4. Requirements: `requirements.txt`
The `requirements.txt` file specifies the Python dependencies necessary to run the notebook and the associated scripts. Key libraries include:
- `transformers==4.41.2`: Provides Hugging Face's transformer models and tokenizers.
- `mamba-ssm`: Required for specialized model layers.
- `causal-conv1d>=1.2.0`: Adds causal convolution layers.
- `accelerate>=0.26.0`: Enables distributed training across multiple GPUs.  

Ensure that you install the required dependencies using:
```bash
pip install -r requirements.txt
```
> Note: The `transformers` version is pinned to avoid compatibility issues with the `Conversation` class (see [Hugging Face Discussion](https://discuss.huggingface.co/t/cannot-import-conversation-from-transformers-utils-py/91556))【8†source】.

## Getting Started

### Prerequisites
Before running the notebook, ensure you have:
- An AWS account with the necessary permissions for using SageMaker.
- Access to Amazon S3 for storing model artifacts.
- SageMaker Studio or a local Jupyter environment with appropriate configurations.

### Steps to Run the Notebook

1. **Set Up the Environment**
   - Open the `JambaRetriever.ipynb` notebook in SageMaker Studio or a Jupyter environment.
   - Run the initial setup cells to configure AWS roles, specify S3 paths, and install any missing dependencies.

2. **Download and Prepare the Dataset**
   - The notebook will guide you through downloading the training dataset (`stsb_multi_mt`) from Hugging Face.
   - Preprocess the dataset into tokenized inputs for the Jamba model.

3. **Fine-Tune the Jamba Model**
   - Train the pre-trained Jamba model using the `train.py` script, running distributed training across multiple GPUs.
   - Training parameters such as batch size, learning rate, and number of epochs can be customized in the notebook.

4. **Deploy the Model for Inference**
   - Once training is complete, deploy the model as a SageMaker endpoint.
   - The notebook will handle the deployment process, using the `inference.py` script to configure the inference behavior.

5. **Test the Deployed Model**
   - Use the provided test cases in the notebook to send text inputs to the deployed model.
   - Evaluate the quality of the embeddings generated for similar and dissimilar sentences.

### Example Usage

Once the endpoint is deployed, you can call it for inference as follows:
```python
predictor = huggingface_model.deploy(
    initial_instance_count=1, 
    instance_type='ml.p3.8xlarge'
)

response = predictor.predict({
    'inputs': "Your input sentence here"
})

print(response)
```

## Future Work

In future iterations, we aim to:
- Optimize the model for lower latency in high-throughput environments.
- Experiment with different loss functions, such as contrastive loss, for better embedding performance.
- Extend support for more input data formats and tasks beyond text similarity.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
