import checklist
from checklist.test_suite import TestSuite
import datasets
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy, compute_accuracy_boolqa
import os
import json
import random
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

NUM_PREPROCESSING_WORKERS = 2


def predconfs(model, context_question_pairs):
    preds = []
    confs = []
    for c, q in context_question_pairs:
        # try:
            p = model(question=q, context=c, truncation=True, )
            preds.append(p['answer'])
            confs.append(p['score'])
        # except Exception:
        #     print('Failed', q)
        #     preds.append(' ')
        #     confs.append(1)
    return preds, np.array(confs)

def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa', 'boolqa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--train_test_split', type=float, default=None,
                      help="""This argument fetches the fraction of test/validation data to train data""")
    argp.add_argument('--analysis', type=str,
                      choices=['contrast_set', 'perturbed_questions', 'adversarial_fine_tuning', 'checklist'], default=None,
                      help="""This argument is used for specifying which analysis we are performing (contrast_set modifies dataset)""")
    argp.add_argument('--checklist_test_suite_path', type=str, default='squad_suite.pkl',
                      help="""This argument specifies the path and filename for a pre-pickled CheckList TestSuite object""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    default_datasets = {'qa': ('squad',), 'nli': ('snli',), 'boolqa': ('boolq',)}
    dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
        default_datasets[args.task]
    analysis_id = tuple(args.analysis.split(':')) if args.analysis is not None else ('na',)
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
    eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
    # Load the raw data
    # Add custom small datasets
    if dataset_id[0] == 'squad':
        dataset = datasets.load_dataset('squad')
        if analysis_id[0] == 'contrast_set':
            # 30 random sample indices
            dataset_samples_idx = [7600, 4206, 253, 5315, 3636, 837, 4301, 6856, 8323, 9668, 6443, 7938, 6899, 8053,
                                   6611,
                                   5171, 7707, 8447, 2531, 5543, 5287, 4938, 7762, 2565, 8135, 7065, 6877, 4010, 3434,
                                   4932]
            dataset = dataset['validation']
            dataset = dataset.select(dataset_samples_idx)
            original_dataset = pd.DataFrame(dataset)
            contrast_set_context = pd.read_csv('contrast_set_context.csv')
            original_dataset['context'] = contrast_set_context['context']
            dataset = datasets.Dataset.from_pandas(original_dataset)
        elif analysis_id[0] == 'adversarial_fine_tuning':
            original_dataset = dataset
            adversarial_dataset = datasets.load_dataset('squad_adversarial', 'AddSent')
            #adversarial_dataset = datasets.load_dataset('squad_adversarial', 'AddOneSent')
            #adversarial_dataset = datasets.concatenate_datasets(
            #    [adversarial_dataset_1['validation'], adversarial_dataset_2['validation']])
            #adversarial_dataset = adversarial_dataset.shuffle(seed=9)
            split_dataset = adversarial_dataset['validation'].train_test_split(args.train_test_split)
            dataset['train'] = datasets.concatenate_datasets([split_dataset['train'], original_dataset['train']])
            dataset['validation'] = split_dataset['test']
        elif analysis_id[0] == 'checklist':
            suite_path = args.checklist_test_suite_path
            suite = TestSuite.from_file(suite_path)

            # Manually build squad-like data from the test suite
            # Put roughly half of it in train and the rest in validation.

            new_data = {
                'train': {'id': [], 'title': [], 'context': [], 'question': [], 'answers': []},
                'validation': {'id': [], 'title': [], 'context': [], 'question': [], 'answers': []}
            }

            num_items = 0

            for test_name in suite.tests:
                test = suite.tests[test_name]

                # Invariance tests do not have labels, so we cannot train against this data
                # the way it is.
                if not test.labels:
                    continue

                # Each test has hundreds of examples.
                # Context/Question pairs are in test.data, and labels are in test.labels
                for pairs, labels in zip(test.data, test.labels):
                    split = 'train' if random.random() > 0.5 else 'validation'

                    # Each series has an equal number of context/question pairs and labels
                    for pair, label in zip(pairs, labels):
                        key = 'checklist%d' % num_items
                        title = 'CheckList example #%d' % num_items
                        num_items += 1

                        new_data[split]['id'].append(key)
                        new_data[split]['title'].append(title)
                        new_data[split]['context'].append(pair[0])
                        new_data[split]['question'].append(pair[1])
                        new_data[split]['answers'].append({
                                'text': [label],
                                'answer_start': [pair[0].index(label)if label in pair[0] else -1]
                            })

            checklist_train_data = datasets.Dataset.from_dict(new_data['train'], features=dataset['train'].features)
            checklist_validation_data = datasets.Dataset.from_dict(new_data['validation'], features=dataset['validation'].features)

            dataset['train'] = datasets.concatenate_datasets([dataset['train'], checklist_train_data])
            dataset['validation'] = datasets.concatenate_datasets([dataset['validation'], checklist_validation_data])

            #dataset[split] = datasets.concatenate_datasets(dataset[split], data)
    elif dataset_id[0] == 'boolq':
        dataset = datasets.load_dataset("super_glue", "boolq")
        if analysis_id[0] == 'perturbed_questions':
            with open('boolq_perturbed.json') as f:
                d = json.load(f)
            perturbed_dataset = json_normalize(d['data'])
            perturbed_dataset.drop(index=perturbed_dataset.index[0], axis=0, inplace=True)
            perturbed_dataset.drop(['title'], axis=1, inplace=True)
            for index, row in perturbed_dataset.iterrows():
                for perturbed_question in row["perturbed_questions"]:
                    perturbed_dataset = perturbed_dataset.append(
                        {'paragraph': row["paragraph"], 'question': perturbed_question['perturbed_q'],
                         'answer': perturbed_question['answer']}, ignore_index=True)
            perturbed_dataset.drop(['perturbed_questions'], axis=1, inplace=True)
            for index, row in perturbed_dataset.iterrows():
                if row['answer'] == 'TRUE':
                    row['answer'] = 1
                else:
                    row['answer'] = 0
            perturbed_dataset.rename(columns={'paragraph': 'passage'}, inplace=True)
            perturbed_dataset.rename(columns={'answer': 'label'}, inplace=True)
            perturbed_dataset['idx'] = perturbed_dataset.index
            dataset = datasets.DatasetDict()
            dataset['train'] = datasets.Dataset.from_pandas(perturbed_dataset)
            dataset['validation'] = datasets.Dataset.from_pandas(perturbed_dataset)
    else:
        dataset = datasets.load_dataset(*dataset_id)

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification,
                     'boolqa': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    elif args.task == 'boolqa':
        boolq_enc = dataset.map(lambda x: tokenizer(x['question'], x['passage'], truncation="only_second"),
                                batched=True)
        prepare_train_dataset = boolq_enc["train"]
        prepare_eval_dataset = boolq_enc["validation"]
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)

    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.task != 'boolqa':
            train_dataset_featurized = train_dataset.map(
                prepare_train_dataset,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=train_dataset.column_names
            )
        else:
            train_dataset_featurized = prepare_train_dataset
    if training_args.do_eval:
        if analysis_id[0] == 'checklist':
            # Just run the checklist test suite and return.
            suite_path = args.checklist_test_suite_path
            suite = TestSuite.from_file(suite_path)
            model_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)

            # Must pass the pairs through a lambda function to use the pre-trained model.
            suite.run(lambda pairs: predconfs(model_pipeline, pairs), overwrite=True)
            suite.summary()
            return
        elif dataset_id[0] == 'squad_mini_30' or dataset_id[0] == 'squad_mini_30_contrast_set':
            eval_dataset = dataset
        else:
            eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.task != 'boolqa':
            eval_dataset_featurized = eval_dataset.map(
                prepare_eval_dataset,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=eval_dataset.column_names
            )
        else:
            eval_dataset_featurized = prepare_eval_dataset

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = datasets.load_metric('squad')
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy
    elif args.task == 'boolqa':
        compute_metrics = compute_accuracy_boolqa

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None

    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()
