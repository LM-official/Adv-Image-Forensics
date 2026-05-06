import  argparse
from    config          import  TEMP_DIR_PATH, MASKED_DIR_NAME, MERGED_METRICS_FILE_NAME, MAX_DESCRIPTION_LENGTH
import  cv2
import  pandas          as      pd
from    pathlib         import  Path
import  sys
import  torch
from    transformers    import  BlipProcessor, BlipForConditionalGeneration
from    utils           import  load_target_metrics


def load_blip() -> tuple[BlipProcessor, BlipForConditionalGeneration, torch.device] | None:
    '''
    Loads the Hugging Face BLIP processor and model.
    '''
        
    print('Loading Hugging Face model (BLIP)...')

    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to load the model and catch any exceptions
    try:
        # check for hardware acceleration (MPS for Apple Silicon, CUDA for Nvidia GPUs) and set the device accordingly
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using hardware acceleration: {device}')

        # load the processor and the pre-trained model for image captioning
        # processor is responsible for preparing the input data (images) in the format expected by the model
        processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        # model is the actual neural network that generates the caption based on the processed input
        model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        
        # move the model to the selected device for faster inference
        model.to(device)
        
        print('BLIP model loaded successfully.')
        return processor, model, device
    
    except Exception as e:
        print(f'ERROR: loading BLIP model: {e}')
        return


def describe_image(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device, masked_image: Path, max_description_len: int) -> str | None:
    '''
    Passes the masked image (numpy array) to a Hugging Face model (BLIP) to get a textual description of the content.
    '''

    print(f'Generating textual description of "{masked_image}"...')
    
    
    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to generate the description and catch any exceptions
    try:
        # prepare the image for the model, Hugging Face processors accept numpy arrays directly (in RGB format)
        image = cv2.imread(str(masked_image))
        if image is None:
            print(f'"{masked_image}" image not found.')
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # move the input tensors to the same device as the model for faster inference
        # optional optimization: process a batch of images together
        inputs = processor(images = image, return_tensors = 'pt').to(device)        
        # generate the prediction
        out = model.generate(**inputs, max_new_tokens = max_description_len)
        # decode the output tensor into a readable text string
        description = processor.decode(out[0], skip_special_tokens = True)

        print(f'Textual description of "{masked_image}" generated successfully: "{description}"')
        return description

    except Exception as e:
        print(f'ERROR: generating description: {e}')
        return



def describe_image_dir(processor: BlipProcessor, model: BlipForConditionalGeneration, device: torch.device, masked_image_dir: Path) -> list[tuple[str, str]] | None:
    '''
    Iterates over all masked images in the director and generates a textual description using the BLIP model.
    '''

    print(f"Processing images in folder: {masked_image_dir}...")


    # LYBL = Look Before You Leap, so we check if the input directory exists before trying to process the images
    if not masked_image_dir.is_dir():
        print(f'ERROR: "{masked_image_dir}" directory not found.')
        return


    descriptions = []
    # iterate on all files
    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to process each image and catch any exceptions (e.g., corrupted files)
    for image in masked_image_dir.iterdir():
        # skip hidden files
        if image.name.startswith('.'):
            continue

        # generate description
        description = describe_image(processor, model, device, image, MAX_DESCRIPTION_LENGTH)
        if description is not None:
            descriptions.append((image.stem, description))


    # sorted by filename
    return sorted(descriptions, key = lambda x: x[0])


def add_description_to_merged_metrics(merged_metrics: Path, descriptions: list[tuple[str, str]]) -> None:
    '''
    Adds the generated descriptions to the merged metrics csv file matching them by filename.
    '''

    print(f'Adding descriptions to merged metrics file: "{merged_metrics}"...')

    # LYBL = Look Before You Leap, so we check if the input file exists and if the descriptions list is not empty before trying to process them
    if not descriptions:
        print('ERROR: descriptions list is empty.')
        return


    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to read the file and catch any exceptions
    try:
        # load merged metrics
        metrics = load_target_metrics(merged_metrics, None)
        if metrics is None or metrics.empty:
            return

        # add descriptions
        print(f'Adding descriptions to "{merged_metrics}"...')
        descriptions = pd.DataFrame(descriptions, columns = ['filename', 'manipulated_description'])
        # how = 'inner' ensures we only keep rows with matching filenames in both dataframes
        metrics = pd.merge(metrics, descriptions, on = 'filename', how = 'inner')

        # reorder: custom order + all remaining columns
        primary_cols = ['filename', 'trufor_score', 'manipulated_description']
        metrics = metrics[primary_cols + [col for col in metrics.columns.to_list() if col not in primary_cols]]


        # save
        # index = False to avoid writing row numbers
        metrics.to_csv(merged_metrics, sep = ',', index = False, encoding = 'utf-8')
        print(f'Descriptions added to "{merged_metrics}" successfully.')

    except Exception as e:
        print(f'ERROR: loading or adding description to "{merged_metrics}": {e}')


if __name__ == '__main__':
    # to visualize an image:
    # python visualize.py --image images/image.png --output output/image.png.npz

    # initialize the argument parser
    parser = argparse.ArgumentParser(description = 'Generate a masked image of the manipulated area and a textual description of it.')    
    
    # flags
    MERGED_METRICS_FILE_PATH  = TEMP_DIR_PATH / MERGED_METRICS_FILE_NAME
    MASKED_DIR_PATH           = TEMP_DIR_PATH / MASKED_DIR_NAME 
    parser.add_argument('--masked_images',      type = Path,    default = MASKED_DIR_PATH,              help = f'Masked images directory path. Default: "{MASKED_DIR_PATH}"')
    parser.add_argument('--merged_metrics',     type = Path,    default = MERGED_METRICS_FILE_PATH,     help = f'Merged metrics csv file path. Default: "{MERGED_METRICS_FILE_PATH}"')
    
    # read the command line arguments
    args = parser.parse_args()


    # load blip model and processor
    blip = load_blip()
    if blip is None:
        sys.exit(1)
    processor, model, device = blip


    # describe images
    descriptions = describe_image_dir(processor, model, device, args.masked_images)


    # update metrics
    add_description_to_merged_metrics(args.merged_metrics, descriptions)