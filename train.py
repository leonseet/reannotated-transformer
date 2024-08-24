import os
import random
import re
import time

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch.nn.functional import pad
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import configs
from modules.model import make_model


def get_device(gpu=None):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device(configs.TRAIN_GPU)  # Automatically use GPU if available, input GPU device number if you want to use a specific GPU


def subsequent_mask(max_seq_len):
    mask = torch.ones(1, max_seq_len, max_seq_len)
    mask = mask.triu(diagonal=0).transpose(-1, -2).type(torch.uint8)
    return mask


def greedy_decode(model, src, src_mask, max_len, start_idx, end_idx):
    ys = torch.tensor(start_idx).to(src.data.dtype).reshape(1, -1).to(DEVICE)
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    for _ in range(max_len - 1):
        x = model.encode(src, src_mask)
        logits = model.decode(x, src_mask, ys, subsequent_mask(len(ys)).to(DEVICE))
        out = model.generator(logits[:, -1])
        _, pred_idx = torch.max(out, dim=-1)
        pred_idx = pred_idx.to(src.data.dtype).reshape(1, -1).to(DEVICE)
        ys = torch.cat([ys, pred_idx], dim=-1)
        if pred_idx == end_idx:
            break
    return ys


def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode="train", accum_iter=1, start_idx=None, end_idx=None, tgt_tokenizer=None, epoch_idx=None):
    assert mode in ["train", "eval"]
    start_time = time.time()
    total_loss = 0
    n_accum = 0
    total_tokens = 0
    tokens = 0
    eos_string = configs.EOS
    pattern = r"(?:<bos>)?(.*?)(?:<eos>)?"

    predicted_target_tokens = []
    gt_target_tokens = []

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train":
            loss_node.backward()
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
            scheduler.step()

        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1 and (mode == "train"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time
            print(
                f"Epoch Step: {i:6d} | Accumulation Step: {n_accum:3d} | Loss: {loss / batch.ntokens:6.2f} | Efficiency: {tokens / elapsed:7.2f} tok/s | Learning Rate: {lr:6.2e}"
            )
            tokens = 0
            start_time = time.time()

        del loss
        del loss_node

    if mode == "eval" and epoch_idx + 1 % 5 == 0 or epoch_idx == 0:
        for i, batch in enumerate(data_iter):
            for idx, (src, src_mask, tgt) in enumerate(zip(batch.src, batch.src_mask, batch.tgt)):
                print(f"Validating: {idx}/{len(batch.src)}")
                y = greedy_decode(model, src.unsqueeze(0), src_mask, 72, start_idx, end_idx)
                predicted_y = tgt_tokenizer.decode(y[0])
                print("Predicted: ", predicted_y)
                predicted_target_tokens.append(predicted_y)
                gt_y = tgt_tokenizer.decode(tgt)
                gt_y = gt_y.split(eos_string)[0] + eos_string
                print("Actual: ", gt_y)
                print()
                gt_target_tokens.append([gt_y])

        gt_target_tokens = [[re.search(pattern, y).group(1) for y in x] for x in gt_target_tokens]
        print("gt_target_tokens: ", gt_target_tokens)
        predicted_target_tokens = [re.search(pattern, x).group(1) for x in predicted_target_tokens]
        print("predicted_target_tokens: ", predicted_target_tokens)
        bleu_score = corpus_bleu(gt_target_tokens, predicted_target_tokens)
        print(f"BLEU-4: {bleu_score}")
        return {"loss": total_loss / total_tokens, "bleu": bleu_score}

    return {"loss": total_loss / total_tokens}


def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (model_size**-0.5 * min(step**-0.5, step * warmup**-1.5))


class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, padding_idx, smoothing):
        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.true_dist = None

    def forward(self, x, target):
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # 2 because one for target and one for padding
        target = target.reshape(-1, 1)
        padding = target.data.clone()
        padding = padding.fill_(self.padding_idx)
        true_dist = true_dist.scatter(-1, target, self.confidence)
        true_dist = true_dist.scatter(-1, padding, 0.0)

        mask = torch.nonzero(target.data.squeeze(-1) == self.padding_idx)

        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(-1), 0.0)

        self.true_dist = true_dist
        loss = self.criterion(x, true_dist)
        return loss


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src.to(DEVICE)
        self.src_mask = (src != pad).unsqueeze(-2).to(DEVICE)
        if tgt is not None:
            self.tgt = tgt[:, :-1].to(DEVICE)
            self.tgt_y = tgt[:, 1:].to(DEVICE)
            self.tgt_mask = self.make_std_mask(self.tgt, pad).to(DEVICE)
            self.ntokens = (self.tgt_y != pad).data.sum().item()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return sloss.data * norm, sloss


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


def get_training_corpus(dataset):
    return (dataset[start_idx : start_idx + 1000] for start_idx in range(0, len(dataset), 1000))


def tokenize_sample(sample, src_tokenizer, tgt_tokenizer):
    src_input_ids = src_tokenizer(sample["de"])["input_ids"]
    tgt_input_ids = tgt_tokenizer(sample["en"])["input_ids"]
    return {"src_input_ids": src_input_ids, "tgt_input_ids": tgt_input_ids}


def collate_batch(batch, src_tokenizer, tgt_tokenizer, device):
    max_src_length = max([len(sample["src_input_ids"]) for sample in batch])
    max_tgt_length = max([len(sample["tgt_input_ids"]) for sample in batch])
    tgt_bos_id = torch.tensor([tgt_tokenizer.bos_token_id], device=device).to(torch.int64)
    tgt_eos_id = torch.tensor([tgt_tokenizer.eos_token_id], device=device).to(torch.int64)

    src_list = []
    tgt_list = []

    for sample in batch:
        src_input_ids = torch.tensor(sample["src_input_ids"], device=device).to(torch.int64)
        tgt_input_ids = torch.tensor(sample["tgt_input_ids"], device=device).to(torch.int64)
        tgt_input_ids = torch.cat([tgt_bos_id, tgt_input_ids, tgt_eos_id])
        src_list.append(pad(src_input_ids, pad=(0, max_src_length - len(src_input_ids)), value=tgt_tokenizer.pad_token_id))
        tgt_list.append(pad(tgt_input_ids, pad=(0, max_tgt_length - len(tgt_input_ids)), value=tgt_tokenizer.pad_token_id))

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    device = f"cuda:{device}"
    return Batch(src, tgt, pad=src_tokenizer.pad_token_id)


class BatchSampler:
    def __init__(self, lengths, batch_size, shuffle=True, drop_last=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        size = len(self.lengths)
        indices = list(range(size))
        if self.shuffle:
            random.shuffle(indices)

        step = min(self.batch_size * 100, size // 4)

        for i in range(0, size, step):
            pool = indices[i : i + step]
            pool = sorted(pool, key=lambda x: self.lengths[x])
            for j in range(0, len(pool), self.batch_size):
                batch = pool[j : j + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    break
                if self.shuffle:
                    random.shuffle(batch)
                yield batch

    def __len__(self):
        return len(self.lengths) // self.batch_size


def load_tokenizers():
    if os.path.exists("tokenizer/en_tokenizer") and os.path.exists("tokenizer/de_tokenizer"):
        src_tokenizer = AutoTokenizer.from_pretrained("tokenizer/en_tokenizer")
        tgt_tokenizer = AutoTokenizer.from_pretrained("tokenizer/de_tokenizer")
        print("Tokenizers loaded from local.")
    else:
        ds = load_dataset("bentrevett/multi30k")
        train_dataset = ds["train"]
        en_corpus = get_training_corpus(train_dataset["en"])
        de_corpus = get_training_corpus(train_dataset["de"])
        old_tokenizer = AutoTokenizer.from_pretrained("gpt2", bos_token=configs.BOS, eos_token=configs.EOS, pad_token=configs.PAD)
        src_tokenizer = old_tokenizer.train_new_from_iterator(en_corpus, vocab_size=configs.EN_VOCAB_SIZE)
        tgt_tokenizer = old_tokenizer.train_new_from_iterator(de_corpus, vocab_size=configs.DE_VOCAB_SIZE)
        src_tokenizer.save_pretrained("tokenizer/en_tokenizer")
        tgt_tokenizer.save_pretrained("tokenizer/de_tokenizer")
        print("Tokenizers built and saved in tokenizer folder.")

    return src_tokenizer, tgt_tokenizer


def create_dataloaders(src_tokenizer, tgt_tokenizer, batch_size=100):
    def tokenize_fn(sample):
        return tokenize_sample(sample, src_tokenizer, tgt_tokenizer)

    def collate_fn(batch):
        return collate_batch(batch, src_tokenizer, tgt_tokenizer, DEVICE)

    # Load Dataset
    ds = load_dataset("bentrevett/multi30k")
    train_dataset = ds["train"]
    validation_dataset = ds["validation"]
    test_dataset = ds["test"]

    # Tokenize Dataset
    train_tokenized_datasets = train_dataset.map(tokenize_fn, batched=True)
    validation_tokenized_datasets = validation_dataset.map(tokenize_fn, batched=True)
    test_tokenized_datasets = test_dataset.map(tokenize_fn, batched=True)

    train_lengths = [len(src) for src in train_tokenized_datasets["src_input_ids"]]

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_tokenized_datasets,
        batch_sampler=BatchSampler(train_lengths, batch_size, shuffle=True, drop_last=False),
        collate_fn=collate_fn,
    )

    subset_size = configs.VALIDATION_SAMPLE_SIZE
    validation_tokenized_datasets = validation_tokenized_datasets.select(range(subset_size))  # Subsetting for faster validation
    validation_dataloader = DataLoader(
        validation_tokenized_datasets,
        batch_size=subset_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_tokenized_datasets,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_dataloader, validation_dataloader, test_dataloader


def train_worker(src_tokenizer, tgt_tokenizer):
    print(f"Train worker process using GPU: {DEVICE} for training", flush=True)
    start_time = time.time()
    pad_idx = src_tokenizer.pad_token_id
    start_idx = src_tokenizer.bos_token_id
    end_idx = src_tokenizer.eos_token_id
    model = make_model(src_tokenizer.vocab_size, tgt_tokenizer.vocab_size, N=configs.N_LAYERS)
    model.to(DEVICE)

    module = model

    criterion = LabelSmoothing(vocab_size=configs.DE_VOCAB_SIZE, padding_idx=pad_idx, smoothing=0.1)
    criterion.to(DEVICE)

    train_dataloader, valid_dataloader, _ = create_dataloaders(
        src_tokenizer,
        tgt_tokenizer,
        batch_size=configs.BATCH_SIZE,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=configs.BASE_LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, configs.D_MODEL, factor=1, warmup=configs.WARMUP),
    )

    results = {"train_loss": [], "valid_loss": [], "bleu-4": []}

    for epoch in range(configs.NUM_EPOCHS):
        model.train()
        print(f"===> [{DEVICE}] Epoch {epoch} Training", flush=True)
        train_results = run_epoch(
            train_dataloader,
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
            accum_iter=configs.ACCUM_ITER,
        )
        results["train_loss"].append(train_results["loss"])
        os.makedirs("models", exist_ok=True)
        if epoch % 5 == 0 or epoch == 0:
            file_path = "models/%s%.2d.pt" % (configs.FILE_PREFIX, epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"===> [{DEVICE}] Epoch {epoch} Validation", flush=True)
        model.eval()
        valid_results = run_epoch(
            valid_dataloader,
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
            start_idx=start_idx,
            end_idx=end_idx,
            tgt_tokenizer=tgt_tokenizer,
            epoch_idx=epoch,
        )
        print(valid_results)
        results["valid_loss"].append(valid_results["loss"])
        results["bleu-4"].append(valid_results["bleu"] if "bleu" in valid_results else None)
        torch.cuda.empty_cache()

    file_path = "models/%sfinal.pt" % configs.FILE_PREFIX
    torch.save(module.state_dict(), file_path)
    time_elapsed = time.time() - start_time
    results["total_training_duration"] = time_elapsed
    results["epoch"] = configs.NUM_EPOCHS
    return results


en_tokenizer, de_tokenizer = load_tokenizers()
results = train_worker(src_tokenizer=de_tokenizer, tgt_tokenizer=en_tokenizer)


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

# Plot 1: Train and Validation Loss
epochs = range(1, len(results["train_loss"]) + 1)
ax1.plot(epochs, results["train_loss"], "b-", label="Train Loss")
ax1.plot(epochs, results["valid_loss"], "g-", label="Validation Loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.set_title("Train and Validation Loss")

# Plot 2: BLEU-4 Score
ax2.plot(epochs, results["bleu-4"], "r-", label="Validation BLEU-4")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("BLEU-4 Score")
ax2.legend()
ax2.set_title("BLEU-4 Score")

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig("training_results.png", dpi=300, bbox_inches="tight")

# Show the plots
# plt.show()
