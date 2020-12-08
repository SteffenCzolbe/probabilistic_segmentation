import pytorch_lightning as pl
from argparse import ArgumentParser
import src.util as util
import torch
import matplotlib.pyplot as plt
import os


def main():

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help=f"Model architecute. Options: {list(util.get_supported_models().keys())}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help=f"Dataset. Options: {list(util.get_supported_datamodules().keys())}",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batchsize. Default 64."
    )
    parser.add_argument(
        "--learning_rate",
        default=0.0001,
        type=float,
        help="Learning rate. Default 0.0001",
    )
    parser.add_argument(
        "--notest", action="store_true", help="Set to not run test after training."
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=100,
        help="Stop active learning after this many iterations.",
    )

    parser.add_argument(
        "--compute_comparison_metrics",
        type=bool,
        default=False,
        help="Compute GED etc. metrics in val and test steps.",
    )

    parser.add_argument(
        "--start_with",
        type=int,
        help='How many samples to start active leanring with'
    )

    parser.add_argument(
        "--add",
        type=int,
        help="How many samples to add after each iteration"
    )

    parser = pl.Trainer.add_argparse_args(parser)
    for model in util.get_supported_models().values():
        parser = model.add_model_specific_args(parser)
    args = parser.parse_args()

    class ResetTrainer:
        def __init__(self, strategy, iter):
            self.checkpointing_callback = pl.callbacks.ModelCheckpoint(
                monitor="val/loss", mode="min", verbose=False
            )
            self.early_stop_callback = pl.callbacks.EarlyStopping(
                monitor="val/loss",
                min_delta=0.00,
                patience=10,
                verbose=False,
                mode="min",
            )
            self.trainer = pl.Trainer.from_argparse_args(
                args,
                checkpoint_callback=self.checkpointing_callback,
                replace_sampler_ddp=False,
                logger=pl.loggers.TensorBoardLogger(
                    save_dir=os.getcwd(),
                    name=f"active_logs/{args.model}/{args.dataset}/{strategy}",
                    version=f"{iter}",
                ),
            )

    # ------------
    # data
    # ------------
    dataset = util.load_damodule(args.dataset, batch_size=args.batch_size)
    args.data_dims = dataset.dims
    args.data_classes = dataset.classes
    dataset.num_workers = 32

    # ------------
    # active_learning
    # ------------
    model_cls = util.get_model_cls(args.model)
    test = {}
    for strategy in ["random", "output_uncertainty"]:
        pl.seed_everything(1234)
        test[strategy] = []
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dataset.augment = None
            size_Q = len(dataset.train_dataloader().dataset)
            mask = torch.randint(size_Q, size=(args.start_with,))
            dataset.sampler = mask.tolist()
        trainer = ResetTrainer(strategy, 0).trainer
        model = model_cls(args)
        trainer.fit(model, dataset)
        test[strategy].append(trainer.test()[0]["test/loss"])
        for i in range(args.num_iters):
            print(f"{strategy}************{i+1}/{args.num_iters}***********")
           
            with torch.no_grad():
                if (i + 1) % 10 == 0:
                    torch.save(
                        test[strategy],
                        f"active_logs/{args.model}/{args.dataset}/{strategy}/loss_{i+1}.pt",
                    )
                    torch.save(
                        test["mask_" + strategy],
                        f"active_logs/{args.model}/{args.dataset}/{strategy}/mask_{i+1}.pt",
                    )

                if strategy == "random":
                    next_query = torch.randint(size_Q, size=(args.add,))
                    mask = torch.cat((mask, next_query))
                elif strategy == "output_uncertainty":
                    model.to(device)
                    output_entropy = []
                    dataset.sampler = None
                    for x, ys in dataset.train_dataloader():
                        x = x.to(device)
                        output_entropy.append(
                            model.pixel_wise_uncertainty(x).sum(dim=(1, 2, 3))
                        )
                    output_entropy = torch.cat(output_entropy)
                    next_query = torch.argsort(output_entropy)[-args.add:]
                    mask = mask.to(device)
                    mask = torch.cat((mask,next_query))
                dataset.sampler = mask.tolist()
            trainer = ResetTrainer(strategy, i + 1).trainer
            model = model_cls(args)
            trainer.fit(model, dataset)
            test[strategy].append(trainer.test()[0]["test/loss"])
        test["mask_" + strategy] = dataset.sampler

    # ------------
    # plot
    # ------------
    print(test)
    plt.figure()
    plt.plot(test["random"], label="random sampling")
    plt.plot(test["output_uncertainty"], label="uncertainty sampling")
    plt.legend()
    plt.xlabel("Iterations of active learning")
    plt.ylabel("Test loss")
    plt.title(f" {args.model} {args.dataset}")
    plt.savefig("plots/active_%s_%s.pdf" % (args.model, args.dataset))


if __name__ == "__main__":
    main()
