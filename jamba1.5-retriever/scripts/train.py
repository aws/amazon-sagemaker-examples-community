import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset


def preprocess_function(examples, tokenizer, max_lenght):
    # Used to tokenize and format datasets
    print("insde the dataset preprocess_function")

    # Tokenize sentence1 and sentence2 separately
    sentence1_inputs = tokenizer(examples['sentence1'], truncation=True, padding=True, max_length=max_lenght)
    sentence2_inputs = tokenizer(examples['sentence2'], truncation=True, padding=True, max_length=max_lenght)

    # Binary labels based on similarity score
    labels = [1 if score >= 2.5 else -1 for score in examples['similarity_score']]

    print("before retuning the dictionary for both sentences")

    # Return a dictionary with input_ids and attention_mask for both sentences, plus labels
    return {
        'input_ids1': sentence1_inputs['input_ids'],
        'attention_mask1': sentence1_inputs['attention_mask'],
        'input_ids2': sentence2_inputs['input_ids'],
        'attention_mask2': sentence2_inputs['attention_mask'],
        'labels': labels
    }


class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    # Custom data collator that handles both input_ids and labels

    def __call__(self, features):
        # Separate input_ids and attention_mask for sentence1 and sentence2
        sentence1_features = [{'input_ids': f['input_ids1'], 'attention_mask': f['attention_mask1']} for f in features]
        sentence2_features = [{'input_ids': f['input_ids2'], 'attention_mask': f['attention_mask2']} for f in features]

        # Call the parent method to handle padding of sentence1 and sentence2
        batch_sentence1 = super().__call__(sentence1_features)
        batch_sentence2 = super().__call__(sentence2_features)

        # Combine sentence1 and sentence2 into a single batch dictionary
        batch = {
            'input_ids1': batch_sentence1['input_ids'],
            'attention_mask1': batch_sentence1['attention_mask'],
            'input_ids2': batch_sentence2['input_ids'],
            'attention_mask2': batch_sentence2['attention_mask']
        }

        batch['labels'] = torch.stack([f['labels'] for f in features])

        return batch


class CustomTrainer(Trainer):

    # Custom loss function based on cosine similarity
    def compute_loss(self, model, inputs, return_outputs=False):

        print("Inside Compute_loss")
        print("inputs: ", inputs)
        print("return_ouputs: ", return_outputs)

        # Extract labels
        labels = inputs["labels"]

        # Extract input_ids and attention_mask for sentence1 and sentence2
        sentence1_input_ids, sentence2_input_ids = inputs['input_ids1'], inputs['input_ids2']
        sentence1_attention_mask, sentence2_attention_mask = inputs['attention_mask1'], inputs['attention_mask2']

        # Pass sentence1 and sentence2 through the model to get embeddings
        outputs1 = model(input_ids=sentence1_input_ids, attention_mask=sentence1_attention_mask)
        print("Outputs1: ", outputs1)

        outputs2 = model(input_ids=sentence2_input_ids, attention_mask=sentence2_attention_mask)
        print("Outputs2: ", outputs2)

        # Pool embeddings
        print("Before mean_pooling call for Embeddings1")
        embeddings1 = mean_pooling(outputs1.last_hidden_state, sentence1_attention_mask)
        print("embeddings1: ", embeddings1)

        print("Before mean_pooling call for Embeddings2") 
        embeddings2 = mean_pooling(outputs2.last_hidden_state, sentence2_attention_mask)
        print("embeddings2: ", embeddings2)

        print("Before constrastive_loss call")
        # Calculate contrastive loss
        loss = contrastive_loss(embeddings1, embeddings2, labels)
        print("After constrastive_loss call")
        print("Return for compute_loss: ", (loss, (embeddings1, embeddings2)))

        if return_outputs:
            print("Return loss and embeddings")
            return (loss, (embeddings1, embeddings2))
        else:
            print("Return only loss")
            return loss

    # Custom Predict Step used during Evaluation steps. Needed to pool embeddings and calculate loss based on contrastive objective
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):

        print("inside prediction_step")
        print(f"inputs: {inputs}, prediction_loss_only: {prediction_loss_only}, ignore-keys: {ignore_keys}")

        # Extract labels
        labels = inputs["labels"]

        print("will call compute_loss to return loss and embeddings")
        # Compute loss and embeddings for 
        with torch.no_grad():
            loss, (embeddings1, embeddings2) = self.compute_loss(model, inputs, return_outputs=True)
        print("after compute_loss")
        print(f"loss , embeddings1, embeddings2: {loss},{embeddings1},{embeddings2}")

        # Return loss, embeddings, and labels for evaluation
        if prediction_loss_only:
            print("retuning loss only inside prediction_step")
            return loss
        else:
            print("retuning loss embeddings and labels inside prediction_step")
            return (loss, (embeddings1, embeddings2), labels)


def mean_pooling(token_embeddings, attention_mask):
    # Mean pooling function to get sentence-level embeddings
    print("inside mean_pooling with token_embeddings")
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def contrastive_loss(embeddings1, embeddings2, labels, margin=0.5):
    # Compute cosine similarity between embeddings1 (sentence 1) and embeddings2 (sentence 2)
    print("Inside Contrastive_loss")
    print("Labels:", labels)

    print(f"embeddings1: {embeddings1}")
    print(f"embeddings2: {embeddings2}")

    cosine_sim = F.cosine_similarity(embeddings1, embeddings2)

    # Positive pairs (similar): Want similarity to be close to 1
    positive_loss = (1 - cosine_sim) * (labels == 1).float()

    # Negative pairs (dissimilar): Want similarity to be low, at least less than the margin
    negative_loss = (cosine_sim - margin).clamp(min=0) * (labels == -1).float()

    # Total loss is the sum of positive and negative losses
    loss = positive_loss + negative_loss

    # Return the mean loss over the batch
    print(f"loss.mean: {loss.mean}")
    print("done with Contrastive_loss")
    return loss.mean()


def compute_metrics(eval_pred, compute_result=False):

    print("Inside compute_metric")
    print(f"eval_pred: {eval_pred}")
    print(f"compute_result: {compute_result}")

    (embeddings1, embeddings2), labels = eval_pred

    # Move tensors to CPU if they are on GPU
    embeddings1 = embeddings1.cpu() if embeddings1.is_cuda else embeddings1
    embeddings2 = embeddings2.cpu() if embeddings2.is_cuda else embeddings2

    # Ensure labels are on CPU and converted to numpy arrays as expected by sklearn. 
    labels = labels.cpu().detach().numpy() if labels.is_cuda else labels.detach().numpy()  

    print("Calculating Cosine Similarity")
    # Calculate cosine similarity between pairs
    cosine_sim = F.cosine_similarity(embeddings1, embeddings2).detach().cpu().numpy()
    print("After Cosine Similarity")
    print(f"cosine_sim: {cosine_sim}")

    print("Calculating Prediction")
    # Convert cosine similarity to binary predictions (1 for similar, -1 for dissimilar)
    predictions = [1 if sim >= 0.5 else -1 for sim in cosine_sim]
    print("After Prediction")
    print(f"predictions: {predictions}")

    # Calculate accuracy, precision, recall, F1 and support score
    print("Calculating Accuracy")
    accuracy = accuracy_score(labels, predictions)
    print("After Accuracy")
    print(f"predictions: {accuracy}")
    print("Calculating Precision, Recall, F1, Support")
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average='binary')
    print("After Precision, Recall, F1, Support")
    print(f"precision: {precision}, recall: {recall}, f1: {f1}")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }

    print(f"Calculated Metrics : {metrics}")

    return metrics


def main():

    # Command-line arguments for hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model_name", type=str, default="ai21labs/AI21-Jamba-1.5-Mini")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--log_dir", type=str, default="/opt/ml/output")
    parser.add_argument("--cache_dir_ds", type=str, default="/opt/ml/dataset_cache")
    parser.add_argument("--cache_dir_model", type=str, default="/opt/ml/model_cache")
    parser.add_argument("--huggingface_token", type=str, default="<myToken>")
    parser.add_argument("--dataset_name", type=str, default="stsb_multi_mt")

    args = parser.parse_args()

    print("Processing Datasets and building Training Configurations" )

    print("load Tokenizer")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir_model, token=args.huggingface_token)
    print("After tokenizer / Before model load")

    print("load Model")
    model = AutoModel.from_pretrained(args.model_name,cache_dir=args.cache_dir_model, token=args.huggingface_token)
    print("After model load / Before dataset load")

    # Load and preprocess dataset
    train_ds, test_ds, dev_ds = load_dataset(args.dataset_name, 'en', split=['train[:20%]','test[:20%]','dev[:20%]'], cache_dir=args.cache_dir_ds)
    train_ds_size = train_ds.num_rows
    test_ds_size = test_ds.num_rows
    dev_ds_size = dev_ds.num_rows
    print(f"After dataset load. # of rows: train {train_ds_size}, test {test_ds_size}, dev {dev_ds_size}")

    print("Tokenizing and formatting Datasets")
    tokenized_train_ds = train_ds.map(lambda examples: preprocess_function(examples, tokenizer, max_lenght=400), batched=True)
    tokenized_test_ds = test_ds.map(lambda examples: preprocess_function(examples, tokenizer, max_lenght=400), batched=True)
    tokenized_dev_ds = dev_ds.map(lambda examples: preprocess_function(examples, tokenizer, max_lenght=400), batched=True)

    print("Here are the first row for each dataset split after processing: Train, Test and Dev")
    print(tokenized_train_ds[0])
    print(tokenized_test_ds[0])
    print(tokenized_dev_ds[0])

    # Define Hugging Face Datasets compatible format
    tokenized_train_ds.set_format(type='torch', columns=['input_ids1', 'attention_mask1', 'input_ids2', 'attention_mask2', 'labels'])
    tokenized_test_ds.set_format(type='torch', columns=['input_ids1', 'attention_mask1', 'input_ids2', 'attention_mask2', 'labels'])
    tokenized_dev_ds.set_format(type='torch', columns=['input_ids1', 'attention_mask1', 'input_ids2', 'attention_mask2', 'labels'])
    print("First rows for each dataset tensors: ")
    print(tokenized_train_ds[0])
    print(tokenized_test_ds[0])
    print(tokenized_dev_ds[0])

    # Initialize the custom data collator
    data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    # Define Step related metrics to drive training loop
    steps_per_epoch = train_ds_size // args.train_batch_size
    num_saves_per_epoch = 2
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.1 * total_steps)

    # Define the training arguments
    training_args = TrainingArguments(

        # Output and Checkpointing
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=steps_per_epoch // num_saves_per_epoch,
        save_total_limit=2,  
        load_best_model_at_end=True,

        # Training Control
        do_train=True,
        do_eval=True,
        do_predict=False,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        max_steps=-1,
        fp16=True,
        gradient_checkpointing=False,
        gradient_accumulation_steps=1,

        # Logging and Reporting
        logging_dir=args.log_dir,
        logging_steps=1,
        logging_first_step=True,
        report_to="tensorboard",

        # Evaluation Control
        evaluation_strategy="steps",
        eval_steps=steps_per_epoch // num_saves_per_epoch,
        eval_accumulation_steps=None,
        batch_eval_metrics=True,

        # Optimization
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        weight_decay=0.01,

        # Model Evaluation
        metric_for_best_model='eval_loss',
        greater_is_better=False,

        # Other
        remove_unused_columns=False,
        label_smoothing_factor=0.0
    )

    # Initialize the Trainer based on Custom Trainer Class. Needed for Compute_loss and Prediction_Step overrides
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_dev_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    print("Starting Training")
    # Train the model
    trainer.train()
    print("Training is done")

    print("Start Final Evaluation")
    trainer.evaluate(eval_dataset=tokenized_test_ds)
    print("Final Evaluation done")

    # Save the model and tokenizers
    print("Saving the Model and Tokenizer")
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_model(args.output_dir)

    print("Model is ready for deployment")


if __name__ == "__main__":
    main()