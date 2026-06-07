import argparse
from   config       import DESCRIPTION_BATCH_SIZE, MAX_DESCRIPTION_TOKENS, MASKED_DIR, FILTERED_DATASET_FILE
import cv2
import pandas       as     pd
from   pathlib      import Path
import torch
from   transformers import BlipProcessor, BlipForConditionalGeneration
from   utils        import update_dataset


def load_blip() -> tuple[BlipProcessor, BlipForConditionalGeneration, torch.device] | None:
    """
    Loads the Hugging Face BLIP processor and model for image captioning.

    Args:
        None.

    Returns:
        A tuple (processor, model, device) ready for inference, or None if loading failed.
    """

    print("Loading Hugging Face model (BLIP)...")
    try:
        # Check for hardware acceleration (MPS for Apple Silicon, CUDA for Nvidia GPUs) and set the device accordingly
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        # Use float32 for better accuracy
        torch_dtype = torch.float32
        print(f"Using device: {device} in {torch_dtype} precision.")

        # Processor is responsible for preparing the input data (images) in the format expected by the model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # Model is the actual neural network that generates the caption based on the processed input
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch_dtype)

        # Move the model to the selected device for faster inference
        model.to(device)

        print("BLIP model loaded successfully.")
        return processor, model, device

    except Exception as e:
        print(f"ERROR: loading BLIP model: {e}")
        return None


def _describe_images_batch(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device,
                           batch: list[Path], max_token: int) -> list[tuple[str, str]] | None:
    """
    Passes a batch of masked images to the BLIP model to get textual descriptions of the manipulated area.

    Args:
        processor:  BLIP processor used to prepare the images for the model.
        model:      BLIP model that generates the captions.
        device:     Torch device on which the model and inputs live (cpu, cuda or mps).
        batch:      List of paths to the masked images in the current batch.
        max_token:  Maximum number of tokens in each generated description.

    Returns:
        A list of (filename_stem, description) tuples, or None on failure or empty batch.
    """


    try:
        loads, names = [], []
        # Load all images in the batch
        for image_path in batch:
            # Force the image to be read in color mode
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"ERROR: '{image_path}' image not found or corrupted.")
                continue
            # Hugging Face processors accept numpy arrays directly in RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            loads.append(image)
            names.append(image_path.stem)
        if not loads:
            return None


        # Process the entire batch of images and move the input tensors to the model's device
        inputs = processor(images=loads, return_tensors="pt").to(device)
        # Generate the predictions
        out = model.generate(**inputs, max_new_tokens=max_token)

        # Decode the output tensors into a list of readable text strings
        batch_descriptions = processor.batch_decode(out, skip_special_tokens=True)
        return list(zip(names, batch_descriptions))

    except Exception as e:
        print(f"ERROR: generating description: {e}")
        return None


def describe_images(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device,
                    masked: Path = MASKED_DIR, dataset: Path = FILTERED_DATASET_FILE,
                    batch_size: int = DESCRIPTION_BATCH_SIZE, max_token: int = MAX_DESCRIPTION_TOKENS) -> None:
    """
    Iterates over all masked images in masked_dir in chunks (batches) and generates textual descriptions.

    Args:
        processor:  BLIP processor used to prepare the images for the model.
        model:      BLIP model that generates the captions.
        device:     Torch device on which the model and inputs live (cpu, cuda or mps).
        masked:     Directory containing the masked images to describe.
        batch_size: Number of images processed simultaneously per batch.
        max_token:  Maximum number of tokens in each generated description.
        dataset:    Path to the dataset CSV to merge the descriptions into.

    Returns:
        None.
    """

    if not masked.is_dir() or not dataset.is_file() or max_token <= 0:
        print(f"ERROR: '{masked}' is not a valid directory or negative token number.")
        return None

    # Get image paths, excluding hidden files
    images = [path for path in masked.iterdir() if not path.name.startswith(".")]
    if not images:
        print(f"WARNING: No valid masked images found in '{masked}'.")
        return None
    len_images = len(images)

    # Generating descriptions
    print(f"Describing masked images in folder: '{masked}'...")


    # No batch
    if batch_size <= 0:
        descriptions = _describe_images_batch(processor, model, device, images, max_token)
        if descriptions:
            update_dataset(dataset, pd.DataFrame(descriptions, columns=["filename", "anomaly_txt"]))
        return None


    descriptions = []
    # Process images in chunks of size = batch_size
    for i in range(0, len_images, batch_size):
        print(f"Processing batch {i // batch_size + 1}/{(len_images + batch_size - 1) // batch_size}...", end="\r")

        # Generate the descriptions for the batch
        batch = images[i:i + batch_size]
        batch_results = _describe_images_batch(processor, model, device, batch, max_token)
        if not batch_results:
            print(f"WARNING: No descriptions generated for batch {i // batch_size + 1}.")
            continue

        descriptions.extend(batch_results)

    update_dataset(dataset, pd.DataFrame(descriptions, columns=["filename", "anomaly_txt"]))
    return None


def main() -> None:
    """
    Loads BLIP and captions every masked image, merging the descriptions into the dataset.

    Args:
        None (the masked dir and dataset default from config.py; batch size and max token are per-run CLI flags).

    Returns:
        None.
    """

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Caption the masked anomaly region of every image with BLIP and merge the result into the dataset.")
    
    #Flags
    # Masked dir and dataset come from config.py (the project backbone); only the batch size and the token cap are exposed as flags
    parser.add_argument("--batch-size", type=int, default=DESCRIPTION_BATCH_SIZE, help=f"Number of images captioned simultaneously. Default: {DESCRIPTION_BATCH_SIZE}")
    parser.add_argument("--max-token",  type=int, default=MAX_DESCRIPTION_TOKENS, help=f"Maximum number of tokens in the generated descriptions. Default: {MAX_DESCRIPTION_TOKENS}")
    args = parser.parse_args()

    # Load the BLIP models
    blip = load_blip()
    if blip is None:
        return None
    processor, model, device = blip

    # Describe the masked images (masked dir and dataset default from config; batch size and max token come from the flags)
    describe_images(processor, model, device, batch_size=args.batch_size, max_token=args.max_token)
    return None

if __name__ == "__main__":
    main()