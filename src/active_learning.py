import pytorch_lightning as pl
from argparse import ArgumentParser
import src.util as util
import torch
import matplotlib.pyplot as plt


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

    parser = pl.Trainer.add_argparse_args(parser)
    for model in util.get_supported_models().values():
        parser = model.add_model_specific_args(parser)
    args = parser.parse_args()

    class ResetTrainer:
        def __init__(self):
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
                callbacks=[self.early_stop_callback],
                logger=False,
            )

    # ------------
    # data
    # ------------
    dataset = util.load_damodule(args.dataset, batch_size=args.batch_size)
    args.data_dims = dataset.dims
    args.data_classes = dataset.classes

    # ------------
    # active_learning
    # ------------
    model_cls = util.get_model_cls(args.model)
    test_loss = {}
    for strategy in ["random", "output_uncertainty"]:
        pl.seed_everything(1234)
        model = model_cls(args)
        test_loss[strategy] = []
        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dataset.augment = None
            size_Q = len(dataset.train_dataloader().dataset)
            mask = torch.randint(size_Q, size=(2,)).tolist()
            dataset.sampler = mask
        trainer = ResetTrainer().trainer
        trainer.fit(model, dataset)
        test_loss[strategy].append(trainer.test()[0]["test/loss"])
        for i in range(args.num_iters):
            print(f"{strategy}************{i+1}/{args.num_iters}***********")
            with torch.no_grad():
                if (i + 1) % 5 == 0:
                    torch.save(
                        test_loss[strategy], f"lightning_models/{strategy}_{i+1}.pt"
                    )
                if strategy == "random":
                    mask.append(int(torch.randint(size_Q, size=(1,))))
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
                    next_query = int(torch.argmax(output_entropy))
                    mask.append(next_query)
                dataset.sampler = mask
            trainer = ResetTrainer().trainer
            trainer.fit(model, dataset)
            test_loss[strategy].append(trainer.test()[0]["test/loss"])
        print(mask)

    # ------------
    # plot
    # ------------
    print(test_loss)
    plt.figure()
    plt.plot(test_loss["random"], label="random sampling")
    plt.plot(test_loss["output_uncertainty"], label="uncertainty sampling")
    plt.legend()
    plt.xlabel("Iterations of active learning")
    plt.ylabel("Test loss")
    plt.title(f" {args.model} {args.dataset}")
    plt.savefig("plots/active_%s_%s.pdf" % (args.model, args.dataset))


if __name__ == "__main__":
    main()
