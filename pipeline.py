"""
ML Pipeline
"""
# Importing necessary libraries and modules
import argparse
from ml import basic_cleaning, train_model, evaluate_model

# Importing necessary libraries and modules
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


"""
Function to execute
"""


def execute(args):
    """
    Execute the pipeline
    """
    if args.steps == "all":
        active_steps = ["basic_cleaning", "train_model", "evaluate_model"]
    else:
        active_steps = args.steps

    if "basic_cleaning" in active_steps:
        logging.info("Basic cleaning started")
        basic_cleaning.run()

    if "train_model" in active_steps:
        logging.info("Train model started")
        train_model.run()

    if "evaluate_model" in active_steps:
        logging.info("Evaluate model started")
        evaluate_model.run()


# Main execution block
if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="ML training pipeline")

    parser.add_argument(
        "--steps",
        type=str,
        choices=["basic_cleaning", "train_model", "evaluate_model", "all"],
        default="all",
        help="Pipeline steps",
    )

    main_args = parser.parse_args()

    execute(main_args)
