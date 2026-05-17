import  argparse
from    config      import  NPZ_DIR_PATH, TEMP_DIR_PATH, TRUTH_FILE_NAME
import  csv
from    pathlib     import  Path


def generate_truth(npz_dir_path: Path, csv_file_path: Path) -> None:
    """
    Generates a csv file with image filenames and empty labels.
    """

    print(f"Generating '{csv_file_path}'...")

    # LYBL = Look Before You Leap, so we check the input before doing anything else
    if not npz_dir_path.is_dir():
        print(f"ERROR: '{npz_dir_path}' directory not found.")
        return

    # Glob search only for .npz files
    # .stem.stem removes the 2 extensions e.g. '.jpg.npz'
    elements = sorted([Path(npz.stem).stem for npz in npz_dir_path.glob("*.npz")])

    # Check if any images were found
    if not elements:
        print(f"WARNING: No .npz file found in '{npz_dir_path}'.")
        return


    # Create the output dir if it doesn"t exist
    # Parents = True creates any necessary parent directories
    # Exist_ok = True means that if the dir already exists, it won"t raise an error
    csv_file_path.parent.mkdir(parents = True, exist_ok = True)

    # Write csv
    with open(csv_file_path, mode = "w", newline = "", encoding = "utf-8") as file:
        writer = csv.writer(file)
        
        # Header
        writer.writerow(["filename", "label"])
        # Elements
        for element in elements:
            writer.writerow([element, ""])

    # Messages to user
    print(f"'{csv_file_path}' generated successfully.")
    print("WARNING: for each entry after ',' append 0 = real or 1 = AI.")


def main() -> None:
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description = "Generate a csv file with image filenames and empty labels.")    
    
    # Flags
    parser.add_argument("--npz",         type = Path,     default = NPZ_DIR_PATH,       help = f"Directory path containing .npz files. Default: '{NPZ_DIR_PATH}'")
    parser.add_argument("--output",      type = Path,     default = TEMP_DIR_PATH,      help = f"Output directory path. Default: '{TEMP_DIR_PATH}'")

    # Read the command line arguments
    args = parser.parse_args()

    # Generate the csv file
    TRUTH_FILE_PATH = args.output / TRUTH_FILE_NAME
    generate_truth(args.npz, TRUTH_FILE_PATH)

if __name__ == "__main__":
    main()