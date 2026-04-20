import runpy
import sys


COMMAND_TO_MODULE = {
    "train": "trainer.train_lora.train",
    "merge": "trainer.train_lora.merge",
    "eval": "trainer.train_lora.evaluate",
}


def main():
    command = "train"
    if len(sys.argv) > 1 and sys.argv[1] in COMMAND_TO_MODULE:
        command = sys.argv.pop(1)
    runpy.run_module(COMMAND_TO_MODULE[command], run_name="__main__")


if __name__ == "__main__":
    main()
