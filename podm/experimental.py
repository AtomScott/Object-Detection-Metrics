# coding=utf-8
# Copyright 2021 Atom Scott
# http://www.apache.org/licenses/LICENSE-2.0
#   __   ____  _____  __  __    ___   ___  _____  ____  ____                          ___
#  /__\ (_  _)(  _  )(  \/  )  / __) / __)(  _  )(_  _)(_  _)     o__        o__     |   |\
# /(__)\  )(   )(_)(  )    (   \__ \( (__  )(_)(   )(    )(      /|          /\      |   |X\
# (__)(__)(__) (_____)(_/\/\_)  (___/ \___)(_____) (__)  (__)     / > o        <\     |   |XX\
#

from podm.utils.converter import (
    coco2bb,
    cvat2bb,
    imagenet2bb,
    labelme2bb,
    openimage2bb,
    text2bb,
    vocpascal2bb,
    yolo2bb,
)
from podm.utils.enumerators import FileFormat


def get_converter(file_format):
    """Gets the appropriate format2bb conversion method.

    Parameters
    ----------
    file_format : FileFormat
        A FileFormat instance representing the format of the file to be converted.

    Returns
    -------
    function
        A function that converts a file to bounding boxes.
    """
    if file_format == FileFormat.IMAGENET:
        return imagenet2bb
    elif file_format == FileFormat.YOLO:
        return yolo2bb
    elif file_format == FileFormat.COCO:
        return coco2bb
    elif file_format == FileFormat.CVAT:
        return cvat2bb
    elif file_format == FileFormat.OPENIMAGE:
        return openimage2bb
    elif file_format == FileFormat.PASCAL:
        return vocpascal2bb
    elif file_format == FileFormat.LABEL_ME:
        return labelme2bb
    elif file_format == FileFormat.ABSOLUTE_TEXT:
        return text2bb
    else:
        raise ValueError("Unknown file format.")
