"""
Recursively remove EXIF metadata from JPEG images without decoding,
re-encoding, resizing, or recompressing the original image data.
"""
import os
import sys
import tempfile
import piexif

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg"}

def remove_exif(file_path: str) -> None:
    with open(file_path, "rb") as file:
        original_data = file.read()  # Read original JPEG bytes
    cleaned_data = piexif.remove(original_data)  # Remove EXIF without re-encoding
    if cleaned_data == original_data:
        return

    directory = os.path.dirname(file_path)
    extension = os.path.splitext(file_path)[1]
    file_stat = os.stat(file_path)  # Preserve original file attributes
    descriptor, temp_path = tempfile.mkstemp(prefix=".exif_clean_", suffix=extension, dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(cleaned_data)
            file.flush()
            os.fsync(file.fileno())  # Ensure data is written to disk
        os.chmod(temp_path, file_stat.st_mode)
        os.replace(temp_path, file_path)  # Atomically replace original file
        os.utime(file_path, ns=(file_stat.st_atime_ns, file_stat.st_mtime_ns))
    except Exception:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        raise

def process_directory(directory: str) -> None:
    directory = os.path.abspath(os.path.expanduser(directory.strip().strip('"').strip("'")))
    if not os.path.isdir(directory):
        print(f"[ERROR] Directory not found: {directory}")
        return

    processed = failed = 0
    for root, directories, files in os.walk(directory):  # Recursively scan subdirectories
        for filename in files:
            if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTENSIONS:
                continue
            file_path = os.path.join(root, filename)
            try:
                remove_exif(file_path)
                processed += 1
                print(f"[OK] {file_path}")
            except Exception as error:
                failed += 1
                print(f"[ERROR] {file_path}: {error}")
    print(f"Completed: {processed} processed, {failed} failed.")

def main() -> None:
    for directory in sys.argv[1:]:  # Process directories passed from command line
        process_directory(directory)
    while True:
        directory = input("Directory: ").strip()
        if directory:
            process_directory(directory)

if __name__ == "__main__":
    main()
