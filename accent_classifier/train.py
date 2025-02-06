import logging
import os
import sys
from typing import Optional
import speechbrain as sb
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import WeightedRandomSampler
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.inference.interfaces import foreign_class
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Load pretrained Wav2Vec embeddings model & pretrained classifier we want to optimize
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53"
)
embedding_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

classifier = foreign_class(
    source="warisqr7/accent-id-commonaccent_xlsr-en-english",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    savedir="pretrained_model",
    run_opts="cpu",  # {"device":"cuda"}  # or "cpu"
)

# c.f. https://github.com/JuanPZuluaga/accent-recog-slt2022/blob/main/CommonAccent/accent_id/train.py

logger = logging.getLogger(__name__)


# Data and Hyperparam configurations for trainer
# mkdir for data folder (speechbrain expects this for training)
# and save off dataframes for training and eval
os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv")
test_df.to_csv("data/test.csv")
eval_df.to_csv("data/valid.csv")
data_folder = "./data"
# loading dataset to the DynamicItemDataset class
# because sb docs says training is much faster if you do
train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
    csv_path="data/train.csv",
    replacements={"data_root": data_folder},
)

config = {
    # Basic configuration
    "seed": 43,
    # Paths and folders
    "data_folder": "/root/.cache/kagglehub/datasets/rtatman/speech-accent-archive/versions/2",
    "csv_prepared_folder": "data/",
    "output_folder": "results/ECAPA-TDNN/43",
    "save_folder": "results/ECAPA-TDNN/43/save",
    "rir_folder": "data/rir_folder",
    "train_log": "results/ECAPA-TDNN/43/train_log.txt",
    # Device and preprocessing
    "device": "cpu",
    "skip_prep": True,
    "avoid_if_longer_than": 35.0,
    # Pretrained embedding module
    "ecapa_tdnn_hub": "speechbrain/spkrec-ecapa-voxceleb/embedding_model.ckpt",
    # Feature parameters
    "n_mels": 80,
    "sample_rate": 16000,
    # Training parameters
    "number_of_epochs": 30,
    "batch_size": 32,
    "n_accents": 21,
    "emb_dim": 192,
    # Batching and workers
    "sorting": "random",
    "dynamic_batching": True,
    "max_batch_len": 600,
    "num_bucket": 200,
    "num_workers": 4,
    # Augmentation and preprocessing
    "apply_augmentation": False,
    "load_pretrained": True,
    # Learning parameters
    "lr": 0.0001,
    # Model architecture details
    "embedding_model_config": {
        "input_size": 80,
        "activation": "torch.nn.LeakyReLU",
        "channels": [1024, 1024, 1024, 1024, 3072],
        "kernel_sizes": [5, 3, 3, 3, 1],
        "dilations": [1, 2, 3, 4, 1],
        "attention_channels": 128,
        "lin_neurons": 192,
    },
    # Optimization details
    "optimizer": "torch.optim.Adam",
    "weight_decay": 0.000002,
    # Learning rate scheduling
    "lr_scheduler": {
        "type": "NewBobScheduler",
        "initial_value": 0.0001,
        "improvement_threshold": 0.0025,
        "annealing_factor": 0.9,
        "patient": 0,
    },
    # Dynamic Batch Sampler Configuration
    "dynamic_batch_sampler": {
        "max_batch_len": 600,
        "max_batch_len_val": 600,
        "num_buckets": 200,
        "shuffle_ex": True,
        "batch_ordering": "random",
        "max_batch_ex": 128,
    },
    # Dataloader options with consistent configurations
    "dataloader_opts": {
        "train": {"batch_size": 32, "num_workers": 4},
        "valid": {"batch_size": 32, "num_workers": 4},
        "test": {"batch_size": 32, "num_workers": 4},
    },
    "train_dataloader_opts": {"batch_size": 32, "num_workers": 4},
    "valid_dataloader_opts": {"batch_size": 32, "num_workers": 4},
}


valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
    csv_path="data/valid.csv",
    replacements={"data_root": data_folder},
)
# We also sort the validation data so it is faster to validate
valid_data = valid_data.filtered_sorted(sort_key="duration")

test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
    csv_path="data/test.csv",
    replacements={"data_root": data_folder},
)
# We also sort the test data so it is faster to validate
test_data = test_data.filtered_sorted(sort_key="duration")

datasets = [train_data, valid_data, test_data]

# Initialization of the label encoder. The label encoder assignes to each
# of the observed label a unique index (e.g, 'accent01': 0, 'accent02': 1, ..)
accent_encoder = sb.dataio.encoder.CategoricalEncoder()


# 2. Define audio pipeline:
@sb.utils.data_pipeline.takes("wav")
@sb.utils.data_pipeline.provides("sig")
def audio_pipeline(wav):
    """Load the signal, and pass it and its length to the corruption class.
    This is done on the CPU in the `collate_fn`."""
    # sig, _ = torchaudio.load(wav)
    # sig = sig.transpose(0, 1).squeeze(1)
    sig, _ = librosa.load(wav, sr=hparams["sample_rate"])
    sig = torch.tensor(sig)
    return sig


sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)


# 3. Define label pipeline:
@sb.utils.data_pipeline.takes("accent")
@sb.utils.data_pipeline.provides("accent", "accent_encoded")
def label_pipeline(accent):
    yield accent
    accent_encoded = accent_encoder.encode_label_torch(accent)
    yield accent_encoded


sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

# 4. Set output:
sb.dataio.dataset.set_output_keys(
    datasets,
    ["id", "sig", "accent_encoded"],
)

# Load or compute the label encoder (with multi-GPU DDP support)
# Please, take a look into the lab_enc_file to see the label to index
# mappinng.
accent_encoder_file = os.path.join(config["save_folder"], "accent_encoder.txt")
os.makedirs(config["save_folder"], exist_ok=True)
accent_encoder.load_or_create(
    path=accent_encoder_file,
    from_didatasets=[train_data],
    output_key="accent",
)

# 5. If Dynamic Batching is used, we instantiate the needed samplers.
train_batch_sampler = None
valid_batch_sampler = None


dynamic_hparams = config["dynamic_batch_sampler"]
num_buckets = dynamic_hparams["num_buckets"]

train_batch_sampler = DynamicBatchSampler(
    train_data,
    dynamic_hparams["max_batch_len"],
    num_buckets=num_buckets,
    length_func=lambda x: x["duration"],
    shuffle=dynamic_hparams["shuffle_ex"],
    batch_ordering=dynamic_hparams["batch_ordering"],
)

valid_batch_sampler = DynamicBatchSampler(
    valid_data,
    dynamic_hparams["max_batch_len_val"],
    num_buckets=num_buckets,
    length_func=lambda x: x["duration"],
    shuffle=dynamic_hparams["shuffle_ex"],
    batch_ordering=dynamic_hparams["batch_ordering"],
)

from speechbrain.utils.parameter_transfer import Pretrainer

# Create experiment directory
sb.create_experiment_directory(
    experiment_directory=config["output_folder"],
)

# Create pretrainer directly
pretrainer = Pretrainer(
    collect_in=config["save_folder"],
    loadables={"embedding_model": config["ecapa_tdnn_hub"]},
    paths={"embedding_model": config["ecapa_tdnn_hub"]},
)

config["pretrainer"] = pretrainer

# we sort training data to speed up training and get better results.
train_data = train_data.filtered_sorted(
    sort_key="duration",
    key_max_value={"duration": hparams["avoid_if_longer_than"]},
)
# when sorting do not shuffle in dataloader ! otherwise is pointless
config["train_dataloader_opts"]["shuffle"] = False


class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# Feature Extraction
compute_features = sb.lobes.features.Fbank(n_mels=80)

# Input Normalization
mean_var_norm_input = sb.processing.features.InputNormalization(
    norm_type="sentence", std_norm=False
)

# Epoch Counter
epoch_counter = sb.utils.epoch_loop.EpochCounter(
    limit=config[
        "number_of_epochs"
    ]  # Using the total number of epochs from your config
)

checkpointer = Checkpointer(
    checkpoints_dir=config["save_folder"],
    recoverables={
        "normalizer_input": mean_var_norm_input,
        "embedding_model": embedding_model,
        "classifier": classifier,
        "counter": epoch_counter,
    },
)

# Then update the config
config["checkpointer"] = checkpointer


class EnhancedAccentID(sb.Brain):
    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN and hparams["apply_augmentation"]:
            # added the False for now, to avoid augmentation of any type
            wavs_noise = self.modules.env_corrupt(wavs, lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            lens = torch.cat([lens, lens], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm_input(feats, lens)

        return feats, lens

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)

        # Compute features, embeddings and output
        feats, lens = self.prepare_features(batch.sig, stage)
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens = inputs

        targets = batch.accent_encoded.data

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hparams["apply_augmentation"]:
            targets = torch.cat([targets, targets], dim=0)
            lens = torch.cat([lens, lens], dim=0)

            if hasattr(self.hparams.lr_annealing, "on_batch_end"):
                self.hparams.lr_annealing.on_batch_end(self.optimizer)

        # get the final loss
        loss = self.hparams.compute_cost(predictions, targets, lens)

        # append the metrics for evaluation
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, targets, lens)

        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error_rate"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        "Initializes the model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none)


def create_weighted_sampler(dataset, accent_key="accent"):
    """Create a weighted sampler to handle class imbalance."""
    accent_list = [item[accent_key] for item in dataset.data.values()]
    class_counts = torch.bincount(
        torch.tensor(
            [dataset.label_encoder.encode_label(accent) for accent in accent_list]
        )
    )
    class_weights = 1.0 / class_counts.float()
    sample_weights = [
        class_weights[dataset.label_encoder.encode_label(accent)]
        for accent in accent_list
    ]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def get_run_opts():
    """Returns the default run options for training."""
    run_opts = {
        "debug": False,  # Set to True for debugging
        "debug_batches": 2,  # Number of batches for debugging
        "debug_epochs": 2,  # Number of epochs for debugging
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_parallel_backend": False,  # Set to True for multi-GPU training
        "distributed_launch": False,  # Set to True for distributed training
        "distributed_backend": "nccl",  # Backend for distributed training
        "find_unused_parameters": False,
        "jit_module_keys": None,  # Set to module names to use JIT
        "auto_mix_prec": False,  # Set to True to use automatic mixed precision
        "max_grad_norm": 5.0,  # For gradient clipping
        "nonfinite_patience": 3,  # How many times to retry if loss is non-finite
        "noprogressbar": False,  # Set to True to disable progress bar
    }
    return run_opts


if __name__ == "__main__":
    run_opts = get_run_opts()
    hparams = config

    hparams.update(
        {
            # Modules configuration
            "modules": {
                "compute_features": compute_features,  # Assuming this is defined earlier
                "embedding_model": embedding_model,  # Your ECAPA-TDNN model
                "mean_var_norm_input": mean_var_norm_input,  # Input normalization
                "classifier": classifier,  # Accent classifier
            },
            # Model as a ModuleList
            "model": torch.nn.ModuleList([embedding_model, classifier]),
            # Loss configuration
            "compute_cost": {
                "type": "LogSoftmaxWrapper",
                "loss_fn": {
                    "type": "AdditiveAngularMargin",
                    "margin": 0.2,
                    "scale": 30,
                },
            },
            # Optimizer configuration
            "lr": 0.0001,
            "opt_class": torch.optim.Adam,
            "weight_decay": 0.000002,
            # Learning rate scheduling
            "lr_annealing": {
                "type": "NewBobScheduler",
                "initial_value": 0.0001,
                "improvement_threshold": 0.0025,
                "annealing_factor": 0.9,
                "patient": 0,
            },
        }
    )

    # Initialize brain object
    aid_brain = EnhancedAccentID(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    aid_brain.fit(
        epoch_counter=epoch_counter,
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Final evaluation
    aid_brain.evaluate(
        test_data,
        min_key="error_rate",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
