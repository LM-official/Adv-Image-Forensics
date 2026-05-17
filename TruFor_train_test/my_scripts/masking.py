import  argparse
from    config      import  IMG_DIR_PATH, NPZ_DIR_PATH, MASK_THRESHOLD, TEMP_DIR_PATH, MASKED_DIR_NAME
import  cv2
import  gc
import  numpy       as      np
from    pathlib     import  Path


def load_anomaly_map(npz_file_path: Path) -> np.ndarray | None:
    """
    Loads the .npz file containing the TruFor anomaly map and returns it as a numpy array.
    """

    print(f"Loading anomaly map from '{npz_file_path}'...")

    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to load the .npz file and catch any exceptions
    try:
        # Use context manager to ensure the file is properly closed after reading
        with np.load(npz_file_path) as npz:
            print("Anomaly map loaded successfully.")
            # Load only anomaly map
            return npz["map"]
    except Exception as e:
        print(f"ERROR: loading anomaly map from '{npz_file_path}': {e}")
        return


def create_masked_image(image_path: Path, anomaly_map: np.ndarray, threshold: float, masked_file_path: Path) -> None:
    """
    Loads the original image, using OpenCV,
    binarizes the TruFor anomaly map
    and applies it as a mask.
    Returns a numpy array (RGB format) with a black background and only the manipulated part visible.
    """
    
    print("Creating masked image...")
    
    # LYBL = Look Before You Leap, so we check if the anomaly map is valid before trying to create the masked image
    if anomaly_map is None or not (0.0 <= threshold <= 1.0):
        print("ERROR: invalid anomaly map or threshold for creating masked image.")
        return


    # EAFP = Easier to Ask for Forgiveness than Permission, so we try to create the masked image and catch any exceptions
    try:
        # Load the original image
        # Cv2 loads images in BGR (Blue, Green, Red) format by default
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Image not found at path: '{image_path}'")
            return


        # Resize the map to match the image pixels (width, height)
        # cv2.resize expects (width, height), while numpy shape is (height, width)
        # INTER_NEAREST prevents OpenCV from creating intermediate gray values at the edges
        anomaly_map = cv2.resize(anomaly_map, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
        # binarization: cv2 requires mask to be uint8 format (0 or 255) for bitwise operations
        binary_mask = (anomaly_map >= threshold).astype(np.uint8) * 255
        
        # Apply the mask using OpenCV"s optimized native function
        masked_img = cv2.bitwise_and(img, img, mask=binary_mask)


        # Create output directory if it doesn"t exist
        masked_file_path.parent.mkdir(parents = True, exist_ok = True)
        # Save
        cv2.imwrite(str(masked_file_path), masked_img)
        print("Masked image created successfully.")

    except Exception as e:
        print(f"ERROR: creating masked image: {e}")
        return


def create_masked_image_dir(image_dir: Path, npz_dir: Path, threshold: float, masked_dir: Path) -> None:
    """
    Creates masked images for all images in the given directory, using the corresponding .npz files in the given directory.
    The masked images are saved in the given output directory.
    """

    print(f"Creating masked images for all images in '{image_dir}'...")

    # LYBL = Look Before You Leap, so we check if the input directories are valid before trying to create the masked images
    if not image_dir.is_dir() or not npz_dir.is_dir():
        print(f"ERROR: invalid input directories for creating masked images.")
        return


    # Create output directory if it doesn"t exist
    masked_dir.mkdir(parents = True, exist_ok = True)

    # Glob search only for .npz files
    for npz in npz_dir.glob("*.npz"):
        # TruFor add .npz to filename
        image_name = npz.stem
        image_path = image_dir / image_name
        MASKED_FILE_PATH = masked_dir / image_name

        # Load the anomaly map
        anomaly_map = load_anomaly_map(npz)
        if anomaly_map is None:
            continue

        # Create the masked image
        create_masked_image(image_path, anomaly_map, threshold, MASKED_FILE_PATH)

        # Force deletion of the anomaly map to free up memory
        del anomaly_map
        # Force garbage collection to free up memory immediately
        gc.collect()
    
    print("Masked images created successfully.")


def main() -> None:
    # To visualize an image:
    # python visualize.py --image images/image.png --output output/image.png.npz

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description = "Generate a masked image of the manipulated area and a textual description of it.")    
    
    # Flags
    parser.add_argument("--images",             type = Path,    default = IMG_DIR_PATH,                 help = f"Images directory path. Default: '{IMG_DIR_PATH}'")
    parser.add_argument("--npz",                type = Path,    default = NPZ_DIR_PATH,                 help = f"Directory path containing .npz files. Default: '{NPZ_DIR_PATH}'")
    parser.add_argument("--threshold",          type = float,   default = MASK_THRESHOLD,               help = f"Binarization threshold. Range: 0.0 to 1.0. Default: {MASK_THRESHOLD}.")
    parser.add_argument("--output",             type = Path,    default = TEMP_DIR_PATH,                help = f"Output directory path. Default: '{TEMP_DIR_PATH}'")
    
    # Read the command line arguments
    args = parser.parse_args()

    # Create the masked images for all images
    create_masked_image_dir(args.images, args.npz, args.threshold, args.output / MASKED_DIR_NAME)

if __name__ == "__main__":
    main()