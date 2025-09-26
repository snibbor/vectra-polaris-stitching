#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import zarr
from numcodecs import Blosc
from tifffile import TiffFile, TiffWriter, xml2dict
import dask.array as da
import xml.etree.ElementTree as ET
import re


# ---------------- Zarr v2/v3 compatibility ----------------
def is_zarr_v3() -> bool:
    try:
        return tuple(int(x) for x in zarr.__version__.split(".", 2)[:2])[0] >= 3
    except Exception:
        return False


if is_zarr_v3():
    from zarr.codecs import Blosc as ZarrBlosc

    def make_group(store_path: str):
        # v3 group
        return zarr.group(store=store_path, overwrite=True)

    def make_array(g, name, shape, chunks, dtype, compressor, fill_value=0):
        """
        Try 'compressor=' first (supported in many v3 builds).
        If that fails, use 'codecs=[...]' which is the canonical v3 style.
        Always use 'chunks=' (not 'chunk_shape=').
        """
        try:
            return g.create_array(
                name=name,
                shape=shape,
                chunks=chunks,  # <-- key change
                dtype=dtype,
                compressor=compressor,  # may work in your env
                fill_value=fill_value,
                overwrite=True,
            )
        except TypeError:
            # Fallback for v3 builds that only accept 'codecs'
            return g.create_array(
                name=name,
                shape=shape,
                chunks=chunks,  # <-- key change
                dtype=dtype,
                codecs=[compressor],  # v3 canonical
                fill_value=fill_value,
                overwrite=True,
            )

    def make_blosc(cname="zstd", clevel=5):
        # v3 codec
        return ZarrBlosc(cname=cname, clevel=clevel)


else:
    from numcodecs import Blosc as NumBlosc

    def make_group(store_path: str):
        return zarr.open_group(store_path, mode="w")

    def make_array(g, name, shape, chunks, dtype, compressor, fill_value=0):
        return g.create_dataset(
            name=name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            fill_value=fill_value,
            overwrite=True,
        )

    def make_blosc(cname="zstd", clevel=5):
        return NumBlosc(cname=cname, clevel=clevel, shuffle=NumBlosc.SHUFFLE)


# ---------------- TIFF helpers & inputs ----------------
def rat(tag):
    if tag is None:
        return None
    v = tag.value
    if isinstance(v, (tuple, list)) and len(v) == 2:
        n, d = v
        return float(n) / float(d) if d else None
    if getattr(v, "shape", None) == (2,):
        n, d = v
        return float(n) / float(d) if d else None
    return float(v)


# the tile metadata has a large offset in the X and Y positions. These are probably stage coordinates and there is a fixed offset from the sample tray to the microscope objective.
def parse_tile_meta(p: Path):
    with TiffFile(str(p)) as tf:
        imgData_s0 = tf.series[0]
        p0 = tf.pages[0]
        tags = p0.tags
        resolution = 10000 / rat(tags.get("XResolution", 1))
        x = (
            int(rat(tags.get("XResolution")) * rat(tags.get("XPosition")))
            if tags.get("XPosition")
            else 0
        )
        y = (
            int(rat(tags.get("YResolution")) * rat(tags.get("YPosition")))
            if tags.get("YPosition")
            else 0
        )
        w = int(tags["ImageWidth"].value)
        h = int(tags["ImageLength"].value)
        series = tf.series[0]
        axes = getattr(series, "axes", None)
        shape = series.shape  # e.g. (C,Y,X) or (Y,X) or (Y,X,C)
        dtype = p0.dtype
        # infer channels
        if axes and "C" in axes:
            csize = shape[axes.index("C")]
        elif len(shape) == 3:
            csize = shape[0]  # assume CYX
        else:
            csize = int(tags.get("SamplesPerPixel", 1).value)

        ch_names = [
            (ET.fromstring(page.description).find("Name").text) for page in imgData_s0
        ]
        colors = [
            (ET.fromstring(page.description).find("Color").text) for page in imgData_s0
        ]

        new_colors = []
        for cl in colors:
            cl = str.split(cl, ",")
            a = 1
            r = int(cl[0])
            g = int(cl[1])
            b = int(cl[2])
            RGBint = int.from_bytes([r, g, b, a], byteorder="big", signed=True)
            new_colors.append(RGBint)

    return dict(
        path=p,
        x=x,
        y=y,
        w=w,
        h=h,
        c=csize,
        axes=axes,
        shape=shape,
        dtype=dtype,
        resolution=resolution,
        ch_names=ch_names,
        ch_colors=new_colors,
    )


def read_tile_cyx(p: Path, c_expected: int) -> np.ndarray:
    with TiffFile(str(p)) as tf:
        s = tf.series[0]
        arr = s.asarray()
        axes = getattr(s, "axes", None)
    if arr.ndim == 2:
        return arr
    if axes:
        amap = {a: i for i, a in enumerate(axes)}
        if all(a in amap for a in "CYX"):
            # Already in CYX format
            pass
        elif all(a in amap for a in "YXC"):
            arr = np.moveaxis(arr, [amap["Y"], amap["X"], amap["C"]], [1, 2, 0])
        elif all(a in amap for a in "YX") and arr.ndim == 3:
            other = [i for i in range(arr.ndim) if i not in (amap["Y"], amap["X"])]
            arr = np.take(arr, 0, axis=other[0])
        else:
            # Default conversion to CYX
            if arr.ndim == 3:
                arr = np.moveaxis(arr, (1, 2, 0), (1, 2, 0))
    else:
        if arr.ndim == 3:
            # Assume input is YXC, convert to CYX
            arr = np.moveaxis(arr, (0, 1, 2), (1, 2, 0))

    # Handle channel padding/truncation
    if arr.ndim == 3 and arr.shape[0] != c_expected:
        c = arr.shape[0]
        if c < c_expected:
            pad = np.zeros((c_expected - c, *arr.shape[1:]), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        else:
            arr = arr[:c_expected, ...]
    return arr


# ---------------- NGFF helpers ----------------
def write_multiscales_zattrs(
    root: zarr.Group, datasets_paths: List[str], axes: List[Dict]
):
    root.attrs["multiscales"] = [
        {
            "version": "0.5",
            "name": "image",
            "axes": axes,
            "datasets": [{"path": p} for p in datasets_paths],
            "type": "org.ngff.multiscales",
        }
    ]


def discover_multiscale_paths(root) -> List[str]:
    m = root.attrs.get("multiscales")
    if isinstance(m, list) and m and isinstance(m[0], dict):
        ds = [d.get("path") for d in m[0].get("datasets", []) if isinstance(d, dict)]
        return [p for p in ds if p in root]
    names = [k for k in root.keys() if hasattr(root.get(k), "shape")]

    def _k(s):
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted(names, key=_k)


def build_ome_levels_and_decimate(z0: zarr.Array, levels: int) -> List[da.Array]:
    a0 = da.from_zarr(z0)  # lazy
    arrs = [a0]
    for _ in range(levels - 1):
        a = arrs[-1]
        # Decimate Y and X dimensions (indices 3 and 4 in TCZYX)
        a = a[..., 0::2, 0::2]
        arrs.append(a)
    return arrs


# ---------------- OME-XML generation functions ----------------
def generate_channels_xml(ch_names: List[str], ch_colors: List[int]) -> str:
    """Generate Channel XML elements for OME-XML."""
    channels_xml = []
    for i, (name, color) in enumerate(zip(ch_names, ch_colors)):
        channels_xml.append(
            f'            <Channel ID="Channel:0:{i}" Name="{name}" Color="{color}" SamplesPerPixel="1"/>'
        )
    return "\n".join(channels_xml)


def generate_ome_xml(
    image_shape: Tuple[int, ...],
    dtype: str,
    resolution: float,
    ch_names: List[str],
    ch_colors: List[int],
) -> str:
    """Generate complete OME-XML string with unified metadata."""
    # Generate channels XML
    channels_xml = generate_channels_xml(ch_names, ch_colors)

    # Determine shape based on dimensionality
    if len(image_shape) == 3:
        # Assuming CYX format
        size_c, size_y, size_x = image_shape
    elif len(image_shape) == 2:
        # YX format (single channel)
        size_y, size_x = image_shape
        size_c = 1
    else:
        raise ValueError(f"Unsupported image shape: {image_shape}")

    # Generate the complete OME-XML
    ome_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
    <Image ID="Image:0" Name="Image0">
        <Pixels ID="Pixels:0" 
        DimensionOrder="XYCZT" 
        Type="float" 
        PhysicalSizeX="{resolution}" 
        PhysicalSizeY="{resolution}" 
        SizeT="1" 
        SizeC="{size_c}" 
        SizeZ="1"
        SizeY="{size_y}" 
        SizeX="{size_x}" 
        >
            {channels_xml}
            <TiffData IFD="0" PlaneCount="{size_c}"/>
        </Pixels>
    </Image>
</OME>"""
    # ome_xml = ome_xml.replace("\n", "").replace("\t", "").replace("  ", "")
    # description = ome_xml.encode()

    return ome_xml


# ---------------- OME-TIFF writer (clone OME-XML; tifffile + dask) ----------------
def write_ome_tiff_from_zarr_tifffile(
    root: zarr.Group,
    out_path: Path,
    ome_xml: str,
    tile: int = 512,
    compression: str = "zlib",
):
    """Stream Zarr pyramid to pyramidal OME-TIFF with custom OME-XML."""
    level_paths = discover_multiscale_paths(root)
    if not level_paths:
        raise ValueError("No multiscale datasets found in Zarr group.")

    darrs = [da.from_zarr(root[p]) for p in level_paths]
    base = darrs[0]
    compr = None if compression == "none" else compression

    def _photo(arr):
        return "rgb" if (arr.ndim == 3 and arr.shape[-1] in (3, 4)) else "minisblack"

    with TiffWriter(str(out_path), bigtiff=True, ome=False) as tif:
        photometric = _photo(base)
        tif.write(
            base,
            description=ome_xml,
            tile=(tile, tile),
            compression=compr,
            photometric=photometric,
            # planarconfig="contig",
            subifds=max(0, len(darrs) - 1),
        )
        for lvl in darrs[1:]:
            tif.write(
                lvl,
                tile=(tile, tile),
                compression=compr,
                photometric=photometric,
                # planarconfig="contig",
                # metadata={"axes": "CYX"},
                subfiletype=1,
            )


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(
        description="Stitch TIFF tiles → OME-Zarr (low RAM), CLONE OME-XML from a source tile (Bio-Formats), then stream to pyramidal OME-TIFF with patched ImageDescription."
    )
    ap.add_argument("inputs", nargs="+")
    ap.add_argument(
        "-o",
        "--output",
        required=True,
        help="Final output path (.ome.tif by default, or .zarr with --output-format zarr).",
    )
    ap.add_argument(
        "--output-format",
        choices=["ome-tiff", "zarr"],
        default="ome-tiff",
        help="Default: ome-tiff.",
    )
    # Zarr build options
    ap.add_argument(
        "--chunksize", type=int, default=1024, help="YX chunk size for Zarr."
    )
    ap.add_argument("--levels", type=int, default=5, help="Number of pyramid levels.")
    ap.add_argument("--compressor", choices=["blosc", "zstd", "lz4"], default="blosc")
    ap.add_argument("--cname", choices=["zstd", "lz4", "zlib"], default="zstd")
    ap.add_argument("--clevel", type=int, default=5)
    # OME-TIFF options
    ap.add_argument("--tiff-tile", type=int, default=512, help="OME-TIFF tile size.")
    ap.add_argument(
        "--tiff-compression",
        default="zlib",
        choices=["zlib", "zstd", "lzma", "jpeg", "lzw", "none"],
        help="OME-TIFF compression.",
    )
    ap.add_argument(
        "--keep-zarr",
        action="store_true",
        help="Keep the intermediate Zarr after writing OME-TIFF.",
    )
    args = ap.parse_args()

    # discover inputs
    files: List[Path] = []
    for s in args.inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(
                q for q in p.iterdir() if q.suffix.lower() in (".tif", ".tiff")
            )
        else:
            files += [Path(s)]
    files = list(dict.fromkeys(files))
    if not files:
        raise SystemExit("No TIFF inputs found.")

    metas = [parse_tile_meta(f) for f in files]

    # canvas and dtype
    W = max(m["x"] + m["w"] for m in metas)
    H = max(m["y"] + m["h"] for m in metas)
    C = max(m["c"] for m in metas) if metas else 1
    W_offset = min(m["x"] for m in metas)
    H_offset = min(m["y"] for m in metas)
    # adjust metas to start at 0,0
    for m in metas:
        m["x"] -= W_offset
        m["y"] -= H_offset
    # adjust canvas size
    W -= W_offset
    H -= H_offset
    H = max(H, 1)
    W = max(W, 1)
    dtype = metas[0]["dtype"] if metas else np.uint16
    print(f"Tiles: {len(metas)} | Canvas: {W}x{H} | Channels: {C} | dtype: {dtype}")

    # zarr store path
    base_out = (
        Path(args.output.replace(".ome", ""))
        if ".ome" in args.output
        else Path(args.output)
    )
    store_path = (
        base_out.with_suffix(".zarr")
        if args.output_format != "zarr" or base_out.suffix.lower() != ".zarr"
        else base_out
    )

    # create zarr and paste tiles using CYX format
    root = make_group(str(store_path))
    compressor = make_blosc(cname=args.cname, clevel=args.clevel)

    # CYX
    shape0 = (C, H, W) if C > 1 else (H, W)
    chunks0 = (
        (C, args.chunksize, args.chunksize)
        if C > 1
        else (args.chunksize, args.chunksize)
    )
    z0 = make_array(root, "0", shape0, chunks0, dtype, compressor, fill_value=0)
    for m in metas:
        arr = read_tile_cyx(m["path"], C)
        h, w = arr.shape[-2:]
        z0[: arr.shape[0], m["y"] : m["y"] + h, m["x"] : m["x"] + w] = arr

    arrs = build_ome_levels_and_decimate(z0, args.levels)
    paths = ["0"]
    for i, a in enumerate(arrs[1:], start=1):
        da.to_zarr(a, str(store_path), component=str(i), overwrite=True, compute=True)
        paths.append(str(i))
    axes = [{"name": "c"}, {"name": "y"}, {"name": "x"}]
    write_multiscales_zattrs(root, paths, axes)

    print(f"OME-Zarr written to: {store_path}")

    if args.output_format == "zarr":
        return

    # stream to OME-TIFF with cloned+patched OME-XML in ImageDescription
    ome_tiff_path = Path(args.output)
    if ome_tiff_path.suffix.lower() not in (".tif", ".tiff", ".ome.tif"):
        ome_tiff_path = ome_tiff_path.with_suffix(".ome.tif")

    # Generate OME-XML with unified metadata
    print("Generating OME-XML with unified metadata...")
    metadata_dict = metas[0]
    resolution = metadata_dict.get("resolution", 0.5)
    ch_names = metadata_dict.get("ch_names", [f"Channel {i}" for i in range(C)])
    ch_colors = metadata_dict.get("ch_colors", [-1] * C)
    image_shape = (C, H, W) if C > 1 else (H, W)
    ome_xml = generate_ome_xml(
        image_shape=image_shape,
        dtype=str(dtype),
        resolution=resolution,
        ch_names=ch_names,
        ch_colors=ch_colors,
    )

    print(
        f"Writing OME-TIFF (tile {args.tiff_tile}, compression {args.tiff_compression}) → {ome_tiff_path}"
    )
    root_ro = zarr.open_group(str(store_path), mode="r")
    write_ome_tiff_from_zarr_tifffile(
        root=root_ro,
        out_path=ome_tiff_path,
        ome_xml=ome_xml,
        tile=args.tiff_tile,
        compression=args.tiff_compression,
    )
    print(f"OME-TIFF written: {ome_tiff_path}")

    if not args.keep_zarr:
        print("Deleting intermediate Zarr...")
        shutil.rmtree(store_path, ignore_errors=True)
        print("Done.")


if __name__ == "__main__":
    main()
