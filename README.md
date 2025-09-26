# Vectra Polaris TIFF Stitching Script

This Python script, `stitch_by_tiff_position_tag_dazarr.py`, is designed to stitch multiple TIFF image tiles into a single, large OME-TIFF or OME-Zarr image. It reads position and metadata from the input TIFF files to correctly place each tile. The script is memory-efficient, utilizing Dask and Zarr for handling large datasets that may not fit into RAM.

## Features

- Stitches TIFF tiles based on X/Y position tags.
- Handles multi-channel images.
- Outputs to pyramidal OME-TIFF (default) or OME-Zarr format.
- Memory-efficient processing using Dask and Zarr.
- Customizable chunking, compression, and pyramid levels.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8+**
- **Conda/Mamba**: This guide uses Conda for environment and package management. If you don't have it, you can install Miniconda or Anaconda.

## Setup Instructions

These instructions will guide you through creating a dedicated Conda environment and installing the necessary Python packages.

### 1. Clone the Repository (if applicable)

If you have this script as part of a Git repository, clone it first:
```bash
git clone https://github.com/your-repo/polaris-stitching.git
cd polaris-stitching
```

### 2. Create and Activate a Conda Environment

It's highly recommended to use a separate Conda environment for this project to avoid conflicts with other packages.

Open your terminal or Anaconda Prompt and run the following commands:

```bash
# Create a new Conda environment named 'stitching_env' with Python 3.9
# You can choose a different Python version (e.g., python=3.10, python=3.11) if preferred.
conda create -n stitching_env python=3.9 -y

# Activate the newly created environment
conda activate stitching_env
```
*On Windows, you might need to use `activate stitching_env` instead.*

### 3. Install Dependencies

The project includes a `requirements.txt` file that lists all the necessary Python libraries. You can install them using `pip` within your activated Conda environment.

```bash
# Install the required packages using pip
pip install -r requirements.txt
```
This will install:
- `numpy`: For numerical operations.
- `zarr`: For chunked, compressed, N-dimensional arrays.
- `numcodecs`: For compression codecs (used by Zarr).
- `tifffile`: For reading and writing TIFF files.
- `dask`: For parallel computing and out-of-core computation.

### 4. Verify Installation (Optional)

You can quickly check if the script is accessible and its dependencies are installed by running its help message:

```bash
python stitch_by_tiff_position_tag_dazarr.py --help
```
This should display the script's command-line arguments and options without any errors.

## How to Use the Script

The script is run from the command line. You need to provide the input TIFF files (or a directory containing them) and specify an output path.

### Basic Syntax

```bash
python stitch_by_tiff_position_tag_dazarr.py [INPUTS] -o [OUTPUT_PATH] [OPTIONS]
```

### Key Arguments

- **`inputs`**: Positional argument(s) for the input TIFF files or directories.
  - You can provide multiple individual `.tif` or `.tiff` files.
  - You can provide one or more directory paths. The script will recursively search for `.tif` or `.tiff` files within these directories.
  - Example: `path/to/tile1.tif path/to/tile2.tif` or `path/to/tiles_directory/`

- **`-o`, `--output`** (Required): The path for the final output file.
  - By default, this will be an OME-TIFF file (e.g., `stitched_image.ome.tif`).
  - If `--output-format zarr` is used, this will be the path to a Zarr store (e.g., `stitched_image.zarr`).

### Common Options

- **`--scale-factor`**: A floating-point number used to multiply the pixel values of each channel in the input images. This is particularly useful for adjusting the intensity of images from specific systems like Vectra Polaris which helps center the dynamic range of values (e.g. from 0.0000-1.0000 to 0.0-1000.0).
  - Default: `1000.0`
  - Example: `--scale-factor 500.0`

- **`--output-format`**: Choose the output format.
  - Choices: `ome-tiff` (default), `zarr`.
  - Example: `--output-format zarr`

- **`--keep-zarr`**: When outputting to OME-TIFF, the script first creates an intermediate Zarr store. By default, this Zarr store is deleted after the OME-TIFF is written. Use this flag to keep it.
  - Example: `--keep-zarr`

- **`--levels`**: Number of resolution pyramid levels to generate (default: 5).
  - Example: `--levels 7`

- **`--tiff-compression`**: Compression for the final OME-TIFF file (default: `zlib`).
  - Choices: `zlib`, `zstd`, `lzma`, `jpeg`, `lzw`, `none`.

For a full list of options, run `python stitch_by_tiff_position_tag_dazarr.py --help`.

### Example Usage

Let's assume you have a directory named `my_tiles` containing all your TIFF tiles, and you want to create a stitched OME-TIFF image named `final_stitch.ome.tif`.

1.  **Make sure your Conda environment is active:**
    ```bash
    conda activate stitching_env
    ```

2.  **Run the script:**
    ```bash
    python stitch_by_tiff_position_tag_dazarr.py my_tiles/ -o final_stitch.ome.tif
    ```

This command will:
- Take all TIFF files from the `my_tiles/` directory.
- Stitch them together.
- Generate a 5-level pyramid.
- Save the result as `final_stitch.ome.tif` with zlib compression.
- Delete the intermediate Zarr store.

**Another Example (Outputting to Zarr):**

If you prefer to output to an OME-Zarr store and keep more intermediate options:

```bash
python stitch_by_tiff_position_tag_dazarr.py my_tiles/ -o output_zarr_store --output-format zarr --levels 6 --cname zstd --clevel 3
```

This will:
- Take all TIFF files from `my_tiles/`.
- Stitch them into an OME-Zarr store located at `output_zarr_store.zarr`.
- Generate a 6-level pyramid.
- Use `zstd` compression with level 3 for the Zarr arrays.

## Output Files

- **OME-TIFF (`.ome.tif`)**: A multi-resolution pyramid TIFF file with OME metadata, viewable in software like ImageJ/Fiji, QuPath, Napari, etc.
- **OME-Zarr (`.zarr` directory)**: A directory-based format containing the image data as chunked Zarr arrays with NGFF (Next Generation File Format) metadata. This is excellent for cloud storage and interactive viewing in tools like Napari.

## Troubleshooting

- **"No TIFF inputs found."**: Ensure the input paths are correct and that the files are indeed `.tif` or `.tiff` files. Check permissions.
- **Memory Issues**: While the script is designed to be memory-efficient, extremely large images or very small chunk sizes can still lead to high memory usage. Try increasing the `--chunksize` value.
- **Import Errors**: Double-check that you have activated the correct Conda environment (`stitching_env`) and that `pip install -r requirements.txt` completed successfully.
