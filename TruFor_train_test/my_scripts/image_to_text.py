import  argparse
from    config          import  TEMP_DIR_PATH, MASKED_DIR_NAME, MERGED_METRICS_FILE_NAME, BATCH_SIZE, MAX_DESCRIPTION_LENGTH
import  cv2
import  pandas          as      pd
from    pathlib         import  Path
import  torch
from    transformers    import  BlipProcessor, BlipForConditionalGeneration
from    utils           import  load_target_metrics


def load_blip() -> tuple[BlipProcessor, BlipForConditionalGeneration, torch.device] | None:
    """
    Loads the Hugging Face BLIP processor and model.
    """
        
    print("Loading Hugging Face model (BLIP)...")

    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to load the model and catch any exceptions
    try:
        # Check for hardware acceleration (MPS for Apple Silicon, CUDA for Nvidia GPUs) and set the device accordingly
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        # Use float32 for better accuracy on CPU, MPS and CUDA
        # Use float16 for lower memory usage and faster inference on MPS/CUDA
        torch_dtype = torch.float32
        print(f"Using device: {device} in {torch_dtype} precision.")

        # Load the processor and the pre-trained model for image captioning
        # Processor is responsible for preparing the input data (images) in the format expected by the model
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # Model is the actual neural network that generates the caption based on the processed input
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype = torch_dtype)
        
        # Move the model to the selected device for faster inference
        model.to(device)
        
        print(f"BLIP model loaded successfully.")
        return processor, model, device
    
    except Exception as e:
        print(f"ERROR: loading BLIP model: {e}")
        return


def describe_image_batch(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device, image_batch: list[Path], max_description_len: int) -> list[tuple[Path, str]] | None:
    """
    Passes a batch of images to a Hugging Face model (BLIP) to get textual descriptions of their content.
    Returns a list of tuples containing the image path and its generated description.
    """

    print(f"Generating textual description for the image batch...")
    
    
    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to generate the description and catch any exceptions
    try:
        images, paths = [], []
        # Load all images in the batch
        for image_path in image_batch:
            # Prepare the image for the model, Hugging Face processors accept numpy arrays directly (in RGB format)
            # Force the image to be read in color mode
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"ERROR: '{image_path}' image not found or corrupted.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            paths.append(image_path)

        # If no valid images were loaded in this batch skip
        if not images:
            return


        # Process the entire batch of images
        # Move the input tensors to the same device as the model for faster inference
        inputs = processor(images = images, return_tensors = "pt").to(device)
        
        # Generate the predictions
        out = model.generate(**inputs, max_new_tokens = max_description_len)
        
        # Decode the output tensors into a list of readable text strings
        batch_descriptions = processor.batch_decode(out, skip_special_tokens = True)

        print(f"Textual description generated successfully for the image batch.")
        return list(zip(paths, batch_descriptions))

    except Exception as e:
        print(f"ERROR: generating description: {e}")
        return


def describe_image_dir(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device, masked_image_dir: Path, batch_size: int) -> list[tuple[str, str]] | None:
    """
    Iterates over all masked images in the directory in chunks (batches) and generates textual descriptions using the BLIP model.
    """

    print(f'Processing images in folder: {masked_image_dir}...')


    # LYBL = Look Before You Leap, so we check if the input directory exists and if the batch size is positive before trying to process the images
    if not masked_image_dir.is_dir() or batch_size <= 0:
        print(f"ERROR: '{masked_image_dir}' is not a valid directory or batch size is not positive.")
        return


    # Get image paths excluding hidden files
    image_paths = [path for path in masked_image_dir.iterdir() if not path.name.startswith(".")]
    if not image_paths:
        print(f"WARNING: No valid images found in '{masked_image_dir}'.")
        return
    

    descriptions = []
    # Process images in chunks of size batch_size
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f'Processing batch {i // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}...')
        
        # Generate description
        batch_results = describe_image_batch(processor, model, device, batch_paths, MAX_DESCRIPTION_LENGTH)
        if batch_results is None or not batch_results:
            print(f"WARNING: No descriptions generated for batch {i // batch_size + 1}.")
            continue

        # Save
        for path, description in batch_results:
            print(f"Textual description of '{path.name}' generated successfully: '{description}'")
            descriptions.append((path.stem, description))

    # Sorted by filename
    return sorted(descriptions, key = lambda x: x[0])


def add_description_to_merged_metrics(merged_metrics: Path, descriptions: list[tuple[str, str]]) -> None:
    """
    Adds the generated descriptions to the merged metrics csv file matching them by filename.
    """

    print(f"Adding descriptions to merged metrics file: '{merged_metrics}'...")

    # LYBL = Look Before You Leap, so we check if the input file exists and if the descriptions list is not empty before trying to process them
    if not descriptions:
        print("ERROR: descriptions list is empty.")
        return


    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to read the file and catch any exceptions
    try:
        # Load merged metrics
        metrics = load_target_metrics(merged_metrics, None)
        if metrics is None or metrics.empty:
            return

        # Add descriptions
        print(f"Adding descriptions to '{merged_metrics}'...")
        descriptions = pd.DataFrame(descriptions, columns = ["filename", "manipulated_description"])
        # How = "inner" ensures we only keep rows with matching filenames in both dataframes
        metrics = pd.merge(metrics, descriptions, on = "filename", how = "inner")

        # Reorder: custom order + all remaining columns
        primary_cols = ["filename", "trufor_score", "manipulated_description"]
        metrics = metrics[primary_cols + [col for col in metrics.columns.to_list() if col not in primary_cols]]


        # Save
        # Index = False to avoid writing row numbers
        metrics.to_csv(merged_metrics, sep = ",", index = False, encoding = "utf-8")
        print(f"Descriptions added to '{merged_metrics}' successfully.")

    except Exception as e:
        print(f"ERROR: loading or adding description to '{merged_metrics}': {e}")


def main() -> None:
    # To visualize an image:
    # python visualize.py --image images/image.png --output output/image.png.npz

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description = "Generate a masked image of the manipulated area and a textual description of it.")    
    
    # Flags
    MERGED_METRICS_FILE_PATH  = TEMP_DIR_PATH / MERGED_METRICS_FILE_NAME
    MASKED_DIR_PATH           = TEMP_DIR_PATH / MASKED_DIR_NAME 
    parser.add_argument("--masked_images",      type = Path,    default = MASKED_DIR_PATH,              help = f"Masked images directory path. Default: '{MASKED_DIR_PATH}'")
    parser.add_argument("--merged_metrics",     type = Path,    default = MERGED_METRICS_FILE_PATH,     help = f"Merged metrics csv file path. Default: '{MERGED_METRICS_FILE_PATH}'")
    parser.add_argument("--batch_size",         type = int,     default = BATCH_SIZE,                   help = f"Number of images to process simultaneously. Default: {BATCH_SIZE}.")

    # Read the command line arguments
    args = parser.parse_args()


    # Load blip model and processor
    blip = load_blip()
    if blip is None:
        return
    processor, model, device = blip


    # Describe images
    descriptions = describe_image_dir(processor, model, device, args.masked_images, args.batch_size)


    # Update metrics
    add_description_to_merged_metrics(args.merged_metrics, descriptions)

if __name__ == "__main__":
    main()