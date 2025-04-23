# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

SF_AUDIO_FILE_EXTENSIONS = {".wav", ".flac", ".ogg"}
FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = {".npy", ".wav", ".flac", ".ogg"}


def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization

    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate


def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization: bool = True,
    mono: bool = True,
    frames: int = -1,
    start: int = 0,
    always_2d: bool = True,
    output_sample_rate: Optional[int] = None,
    normalize_volume: bool = False,
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit WAV/FLAC/OGG Vorbis audio.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): normalize values to [-1, 1] (Default: True)
        mono (bool): convert multi-channel audio to mono-channel one
        frames (int): the number of frames to read. (-1 for reading all)
        start (int): Where to start reading. A negative value counts from the end.
        always_2d (bool): always return 2D array even for mono-channel audios
        output_sample_rate (Optional[int]): output sample rate
        normalize_volume (bool): normalize volume
    Returns:
        waveform (numpy.ndarray): 1D or 2D waveform (channels x length)
        sample_rate (float): sample rate
    """
    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext not in SF_AUDIO_FILE_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile: pip install soundfile")

    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
    )
    waveform = waveform.T  # T x C -> C x T
    waveform, sample_rate = convert_waveform(
        waveform,
        sample_rate,
        normalize_volume=normalize_volume,
        to_mono=mono,
        to_sample_rate=output_sample_rate,
    )

    if not normalization:
        waveform *= 2**15  # denormalized to 16-bit signed integers
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform, sample_rate


def get_features_or_waveform(path: str, use_sample_rate=None):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.
        use_sample_rate (int): change sample rate for the input wave file

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    _path, slice_ptr = parse_path(path)
    assert len(slice_ptr) == 0

    return get_waveform(
        _path, always_2d=False, output_sample_rate=use_sample_rate
    )[0]


def read_legacy_audio(path, sample_rate, normalize):
    wav = get_features_or_waveform(path, use_sample_rate=sample_rate)

    if wav.ndim == 2:
        wav = wav.mean(-1)

    assert wav.ndim == 1, wav.ndim
    '''
    if ref_len is not None and abs(ref_len - len(wav)) > 160:
        logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
    '''

    orig_shape = wav.shape

    # now what to do about cuda?
    wav = torch.from_numpy(wav).float().cuda()

    if normalize:
        wav = F.layer_norm(wav, wav.shape)
    wav = wav.view(1, -1)

    assert orig_shape[0] == wav.shape[1], (orig_shape, wav.shape)

    return wav


def parse_path(path: str) -> Tuple[str, List[int]]:
    """Parse data path which is either a path to
    1. a .npy/.wav/.flac/.ogg file
    2. a stored ZIP file with slicing info: "[zip_path]:[offset]:[length]"

      Args:
          path (str): the data path to parse

      Returns:
          file_path (str): the file path
          slice_ptr (list of int): empty in case 1;
            byte offset and length for the slice in case 2
    """

    if Path(path).suffix in FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS:
        _path, slice_ptr = path, []
    else:
        _path, *slice_ptr = path.split(":")
        if not Path(_path).is_file():
            raise FileNotFoundError(f"File not found: {_path}")
    assert len(slice_ptr) in {0, 2}, f"Invalid path: {path}"
    slice_ptr = [int(i) for i in slice_ptr]
    return _path, slice_ptr
