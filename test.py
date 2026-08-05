"""Evaluate the released DGNet checkpoints."""

from train import DGNetTrainer, build_parser, prepare_options


def main():
    options = prepare_options(build_parser(default_mode="test").parse_args())
    options.mode = "test"
    trainer = DGNetTrainer(options)
    try:
        trainer.inference()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
