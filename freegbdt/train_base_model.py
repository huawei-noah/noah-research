# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import numpy as np
from joblib import dump
from torch.utils import data
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import torch
import argparse
from hashlib import md5
from datetime import datetime
from pathlib import Path
import json
import sys

import datasets
from models import ModelForSequenceClassification
import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=False, default=None)
    parser.add_argument(
        "--task", type=str, required=True,
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tqdm", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load_head", default=False, action="store_true")

    hparams = {
        "head": "multilayer",  # 'linear' or 'multilayer'
        "batch_size": 4,
        "grad_acc_steps": 8,
        "max_seq_length": 256,
        "epochs": 10,
        "hidden_dim": 1024,
        "learning_rate": 1e-5,
    }

    for key, value in hparams.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))

    args = parser.parse_args()
    args.device = torch.device(args.device)

    if args.seed is None:
        args.seed = np.random.randint(0, 100_000)

    param_str = ""
    for key, value in args.__dict__.items():
        param_str += f"{key}={value}"

    args.run_hash = (
        args.task
        + "_"
        + datetime.now().strftime("%d%m%H%M")
        + "_"
        + md5(param_str.encode("utf-8")).hexdigest()[:6]
    )
    args.log_dir = (Path(__file__) / "..").resolve() / "logs" / args.run_hash
    args.log_dir.mkdir(parents=True)

    sys.stdout = open(args.log_dir / "out.log", "a")
    sys.stderr = sys.stdout

    print("Parameters:")
    for key, value in args.__dict__.items():
        print(f"{key}={value}", flush=True)

    return args


def evaluate(model, val_loader, unshuffled_train_loader):
    result = get_predictions(model, val_loader)
    extraction_result = get_predictions(model, unshuffled_train_loader)

    if n_labels == 1:
        preds = result[0][:, 0] > 0.5
    else:
        preds = np.argmax(result[0], 1)

    print("valid accuracy:", (preds == val_data["label"].values).mean(), flush=True)

    epoch_val_features = np.concatenate([result[1], result[0]], 1)
    epoch_extracted_features = np.concatenate(
        [extraction_result[1], extraction_result[0]], 1
    )

    return epoch_val_features, epoch_extracted_features


if __name__ == "__main__":
    args = get_args()
    utils.seed_everything(args.seed)

    json.dump(
        {key: str(value) for key, value in args.__dict__.items()},
        open(args.log_dir / "params.json", "w"),
        indent=4,
    )

    dataset = datasets.DATASETS[args.task]

    n_labels = dataset.get_n_labels()
    train_data, val_data, test_data = dataset.load()

    model = ModelForSequenceClassification(
        AutoModel.from_pretrained(str(args.model_path)),
        hidden_dim=args.hidden_dim,
        n_labels=n_labels,
        head=args.head,
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))

    if args.load_head:
        assert args.head == "multilayer"

        x = AutoModelForSequenceClassification.from_pretrained(str(args.model_path))
        model.classifier.load_state_dict(x.classifier.state_dict())

    train_dataset = dataset(tokenizer, train_data, max_length=args.max_seq_length)
    val_dataset = dataset(tokenizer, val_data, max_length=args.max_seq_length)
    test_dataset = dataset(tokenizer, test_data, max_length=args.max_seq_length)

    train_loader = data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    unshuffled_train_loader = data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=val_dataset.collate_fn,
    )
    test_loader = data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=val_dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=int(len(train_loader) / args.grad_acc_steps * args.epochs),
    )

    def get_predictions(model, loader):
        predictions = []
        labels = []
        features = []

        for batch in tqdm(loader, disable=not args.tqdm):
            input_ids = batch[0][0].to(args.device)
            attention_mask = batch[0][1].to(args.device)

            try:
                labels.append(batch[1][0].numpy())
            except IndexError:
                pass

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs[0]
            features.append(outputs[1].detach().cpu().numpy())
            if n_labels == 1:
                batch_predictions = torch.sigmoid(logits).detach().cpu().numpy()
            else:
                batch_predictions = torch.softmax(logits, 1).detach().cpu().numpy()

            predictions.append(batch_predictions)

        out = [np.concatenate(predictions, 0), np.concatenate(features, 0)]

        if len(labels) > 0:
            out.append(np.concatenate(labels, 0))

        return out

    all_train_features = np.zeros(
        (args.epochs, len(train_dataset), args.hidden_dim + n_labels)
    )
    all_train_labels = np.zeros((args.epochs, len(train_dataset)))
    all_extracted_train_features = np.zeros(
        (args.epochs + 1, len(train_dataset), args.hidden_dim + n_labels)
    )
    all_val_features = np.zeros(
        (args.epochs + 1, len(val_dataset), args.hidden_dim + n_labels)
    )

    model.eval()

    epoch_val_features, epoch_extracted_features = evaluate(
        model, val_loader, unshuffled_train_loader
    )

    all_extracted_train_features[0] = epoch_extracted_features
    all_val_features[0] = epoch_val_features

    for epoch in tqdm(range(args.epochs), desc="Epoch", disable=not args.tqdm):
        print(f"Epoch {epoch}", flush=True)

        epoch_train_loss = 0
        model.train()
        model.zero_grad()

        epoch_features = []
        epoch_labels = []

        for step, batch in enumerate(tqdm(train_loader, disable=not args.tqdm)):
            input_ids = batch[0][0].to(args.device)
            attention_mask = batch[0][1].to(args.device)
            labels = batch[1][0].to(args.device)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            outputs = model(**inputs)
            loss = outputs[0]
            loss = loss / args.grad_acc_steps

            epoch_features.append(
                np.concatenate(
                    [
                        outputs[2].detach().cpu().numpy(),
                        outputs[1].detach().cpu().numpy(),
                    ],
                    1,
                )
            )
            epoch_labels.append(labels.detach().cpu().numpy())

            epoch_train_loss += loss.item()
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                model.zero_grad()

        model.eval()

        epoch_val_features, epoch_extracted_features = evaluate(
            model, val_loader, unshuffled_train_loader
        )
        epoch_features = np.concatenate(epoch_features, 0)
        epoch_labels = np.concatenate(epoch_labels, 0)

        all_extracted_train_features[epoch + 1] = epoch_extracted_features
        all_train_features[epoch] = epoch_features
        all_train_labels[epoch] = epoch_labels
        all_val_features[epoch + 1] = epoch_val_features

    test_result = get_predictions(model, test_loader)
    test_features = np.concatenate([test_result[1], test_result[0]], 1)

    dump(
        {
            "train_features": all_train_features,
            "extracted_train_features": all_extracted_train_features,
            "train_labels": all_train_labels,
            "val_features": all_val_features,
            "test_features": test_features,
        },
        args.log_dir / "features.joblib",
    )
