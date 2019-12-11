# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for visualization
"""
import math
import random
from typing import Dict
from typing import List
from typing import Optional

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np


def draw_keypoints_connections(
        image: np.ndarray,
        keypoints: np.ndarray,
        connection_map: Dict[str, list],
        colors: Optional[Dict[str, list]] = None,
        thickness: int = 2,
        add_legend: bool = True,
) -> np.ndarray:
    """
    Draw connections between keypoints according to connection map

    If add_legend is True, then it will add the legend to an image,
    so image width will be changed

    Parameters
    ----------
    image
        np.uint8 image
    keypoints
        keypoints with shape [num_objects, num_keypoints, 2] in global, e.g.
        image coordinates and coordinates in format [y, x]
    connection_map
        connection map is a dict which maps the name of connection to the
        connection indices, e.g. {"joint1": [1, 2]}
    colors
        dict with mapping of the same connection names as connection_map
        to the rgb colors
    thickness
        thickness of the connection
    add_legend
        if the legend should be added to the right side of the image

    Returns
    -------
    image
        image with connections drawn
    """
    num_objects = keypoints.shape[0]
    colors = colors or _create_random_connections_colors(connection_map)
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    for each_object_i in range(num_objects):
        draw_keypoints_connections_single_object(
            draw, keypoints[each_object_i], connection_map,
            colors, thickness=thickness)
    np.copyto(image, np.array(image_pil))
    if not add_legend:
        return image

    image = add_legends(
        image, ["skeleton"], [colors], orientation="vertical")
    return image


def draw_keypoints_connections_single_object(
        draw: ImageDraw.ImageDraw,
        keypoints: np.ndarray,
        connection_map: Dict[str, list],
        colors: Dict[str, list],
        thickness: int = 1,
):
    """
    Draw connections between keypoints according to connection map

    Parameters
    ----------
    draw
        image draw
    keypoints
        keypoints with shape [num_keypoints, 2] in global, e.g.
        image coordinates and coordinates in format [y, x]
    connection_map
        connection as list of lists of connections specifying connected
        keypoints indices
    colors
        list of rgb values for each connection index
    thickness
        thickness of the connection

    Returns
    -------
    image
        image with connections drawn
    """
    valid_keypoints = np.any(keypoints > 0, -1)
    for each_connection_name in connection_map:
        connection_start, connection_end = connection_map[each_connection_name]
        color = colors[each_connection_name]
        is_valid_connection = np.all(
            [valid_keypoints[connection_start],
             valid_keypoints[connection_end]])
        if not is_valid_connection:
            continue
        coordinates = np.concatenate(
            [keypoints[connection_start][::-1],
             keypoints[connection_end][::-1]], 0
        ).astype(np.int32).tolist()
        draw.line(coordinates, fill=tuple(color), width=thickness)


def draw_attributes_as_marks(
        image: np.ndarray,
        object_boxes,
        attributes: Dict[str, np.ndarray],
        class_ids_to_names_mapping: Dict[str, Dict[int, str]],
        colors: Optional[Dict[str, Dict[int, list]]] = None,
        thickness: int = 5,
        add_legend: bool = True,
):
    """
    Draws the small marks on the right or on the left of bounding boxes with
    colors corresponding to the attribute value and adds a legend for attributes
    to an image

    If add_legend is True, then it will add the legend to an image,
    so image width will be changed

    Parameters
    ----------
    image
        image to draw on
    object_boxes
        object boxes with shape [num_objects, 4] and coordinates in format
        [ymin, xmin, ymax, xmax] represented as absolute coordinates
    attributes
        dict which holds the attributes for all objects with every value
        of shape [num_objects]; attributes with value <=0 will be ignored
    class_ids_to_names_mapping
        mapping of type {attribute_name: {class_id: class_name}}
    colors
        colors as mypping of type {attribute_name: {class_id: rgb}}
    thickness
        thickness of the mark
    add_legend
        if the legend should be added to the right side of the image

    Returns
    -------
    image_with_attributes_and_legend
        image with attributes as marks and corresponding legend
    """
    # pylint: disable=too-many-locals
    # TODO(oleksandr.vorobiov@audi.de): refactor
    difference_in_attributes_and_mapping = set(attributes).symmetric_difference(
        set(class_ids_to_names_mapping))
    assert not difference_in_attributes_and_mapping, (
        "attributes must have the same keys as class_ids_to_names_mapping")
    colors = (colors
              or _create_random_attributes_colors(class_ids_to_names_mapping))
    assert not set(attributes).symmetric_difference(set(colors)), (
        "attributes must have the same keys as colors")
    num_objects = object_boxes.shape[0]
    valid_boxes = np.any(
        (object_boxes[..., 2:] - object_boxes[..., :2]) > 0, -1)

    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    for each_object_i in range(num_objects):
        if not valid_boxes[each_object_i]:
            continue

        _draw_attribute_marks(image, draw, attributes, object_boxes,
                              each_object_i, colors, thickness)

    np.copyto(image, np.array(image_pil))
    class_names_to_colors = _get_class_names_to_colors(
        colors, class_ids_to_names_mapping)

    if not add_legend:
        return image

    legend_titles, legend_items_with_colors = zip(
        *sorted(class_names_to_colors.items()))
    result = add_legends(image, legend_titles, legend_items_with_colors,
                         orientation="vertical", legend_markers="box")
    return result


def add_legends(image: np.ndarray,
                legend_titles: List[str],
                legend_items_with_colors: List[Dict[str, list]],
                legend_markers: str = "line",
                orientation="vertical") -> np.ndarray:
    """
    Add legends to the image by stacking them in the direction of orientation

    Will add the legend in vertical (to the right of image) or in horizontal
    (to the bottom of image) orientations.

    Parameters
    ----------
    image
        image
    legend_titles
        list of legends titles
    legend_items_with_colors
        list of dict with mapping of legend items to its RGB colors;
        inside of legend items are sorted in alphabetical order
    orientation
        orientation of the legend - horizontal or vertical
    legend_markers
        which marks to use as a legend markers

    Returns
    -------
    image_with_legend
        image with legend
    """
    assert len(legend_titles) == len(legend_items_with_colors), (
        "length of legend titles must be equal to lengths if legend items!"
    )
    assert legend_markers in ["box", "line"], (
        "currently supported only box and line legend markers"
    )
    legend_images = []
    for each_title, each_items in zip(legend_titles, legend_items_with_colors):
        if orientation == "vertical":
            legend_image = _draw_vertical_legend(
                each_title, each_items, legend_markers=legend_markers)
        else:
            legend_image = _draw_horizontal_legend(
                each_title, each_items, legend_markers=legend_markers)
        legend_images.append(legend_image)

    if len(legend_images) > 1:
        legend_complete = _concat_images(legend_images, orientation)
    else:
        legend_complete = legend_images[0]
    result = _concat_images(
        [image, legend_complete],
        "horizontal" if orientation == "vertical" else "vertical")
    return result


def _get_class_names_to_colors(colors, class_ids_to_names_mapping):
    class_names_to_colors = {}
    for each_name, each_class_id_to_name_map in (
            class_ids_to_names_mapping.items()):
        class_names_to_colors[each_name] = {}
        for each_class_id, each_class_name in each_class_id_to_name_map.items():
            class_names_to_colors[each_name][each_class_name] = (
                colors[each_name][each_class_id])
    return class_names_to_colors


def _draw_attribute_marks(image, draw, attributes, object_boxes, object_index,
                          colors, thickness):
    # pylint: disable=too-many-locals
    # TODO(oleksandr.vorobiov@audi.de): refactor
    image_width = image.shape[1]
    object_box = object_boxes[object_index]
    box_ymin, box_xmin, box_ymax, box_xmax = object_box
    box_height = box_ymax - box_ymin

    attributes_to_draw, attributes_to_draw_colors = (
        _select_valid_attributes_and_colors(attributes, colors, object_index))

    num_attributes_to_draw = len(attributes_to_draw)

    max_marks_per_height = box_height // thickness
    marks_per_width = math.ceil(num_attributes_to_draw / max_marks_per_height)
    marks_width = marks_per_width * thickness
    draw_to_the_right = True
    if box_xmax + marks_width > image_width:
        draw_to_the_right = False

    mark_y = box_ymin
    if draw_to_the_right:
        mark_x = box_xmax
    else:
        mark_x = box_xmin - thickness

    for each_attribute_name in sorted(attributes_to_draw):
        color = attributes_to_draw_colors[each_attribute_name]
        draw.rectangle([mark_x, mark_y, mark_x + thickness, mark_y + thickness],
                       fill=color)

        mark_y += thickness
        if mark_y > box_ymax:
            mark_y = box_ymin
            if draw_to_the_right:
                mark_x += thickness
            else:
                mark_x -= thickness


def _select_valid_attributes_and_colors(attributes, colors, object_index):
    attributes_to_draw = {}
    attributes_to_draw_colors = {}
    for each_attribute_name, each_attribute in attributes.items():
        each_attribute_i = each_attribute[object_index]
        if each_attribute_i <= 0:
            continue
        attributes_to_draw[each_attribute_name] = each_attribute_i
        color = tuple(colors[each_attribute_name][each_attribute_i])
        color = color if isinstance(color, str) else tuple(color)
        attributes_to_draw_colors[each_attribute_name] = color
    return attributes_to_draw, attributes_to_draw_colors


def _concat_images(images,
                   orientation: str, background: tuple = (255, 255, 255)
                   ) -> np.ndarray:
    if orientation == "horizontal":
        new_size = (sum(each_image.shape[1] for each_image in images),
                    max(each_image.shape[0] for each_image in images))
    else:
        new_size = (max(each_image.shape[1] for each_image in images),
                    sum(each_image.shape[0] for each_image in images))
    result_image = Image.new("RGB", new_size, background)
    x_coord, y_coord = 0, 0
    for each_image in images:
        result_image.paste(Image.fromarray(np.uint8(each_image)).convert('RGB'),
                           (x_coord, y_coord))
        if orientation == "vertical":
            y_coord += each_image.shape[0]
        else:
            x_coord += each_image.shape[1]
    return np.array(result_image)


def _draw_vertical_legend(legend_title, legend_items_with_colors,
                          background: tuple = (255, 255, 255),
                          legend_markers: str = "line") -> np.ndarray:
    # pylint: disable=too-many-locals
    # TODO(oleksandr.vorobiov@audi.de): refactor
    indicator_width = 10
    font = _get_font()
    title_width, title_height = font.getsize(legend_title)
    height_offset = title_height // 2
    max_width, total_height = _get_legend_dimensions(
        legend_items_with_colors, title_height, title_width,
        height_offset, indicator_width, font)
    legend_image_pil = Image.new("RGB", [max_width, total_height],
                                 color=background)
    draw = ImageDraw.Draw(legend_image_pil)

    y_coord = title_height + height_offset
    for each_color_name, each_color in sorted(legend_items_with_colors.items()):
        _draw_item_on_legend(draw, each_color, each_color_name, y_coord,
                             indicator_width, legend_markers, font)
        y_coord += height_offset + font.getsize(each_color_name)[1]

    draw.rectangle([2, 2, max_width - 2, total_height - 2], outline=(0, 0, 0))
    draw.rectangle([2, 0, title_width + 7, title_height], background)
    draw.text([2, 0], legend_title, (0, 0, 0), font)
    return np.array(legend_image_pil)


def _get_legend_dimensions(legend_items_with_colors, title_height, title_width,
                           height_offset, indicator_width, font):
    color_widths, color_heights = zip(
        *[font.getsize(each_name) for each_name in legend_items_with_colors])
    max_width = int(max(
        title_width + indicator_width * 1.5,
        *[each_width + indicator_width * 1.5 for each_width in color_widths]))
    max_width += 10
    total_height = (title_height * 2
                    + sum(color_heights) + height_offset * len(color_heights))
    return max_width, total_height


def _draw_item_on_legend(draw, each_color, each_color_name, y_coord,
                         indicator_width, legend_markers, font):
    y_indicator = y_coord + font.getsize(each_color_name)[1] // 2
    y_text = y_coord
    each_color = (tuple(each_color) if not isinstance(each_color, str)
                  else each_color)
    if legend_markers == "line":
        draw.line([5, y_indicator, 5 + indicator_width, y_indicator],
                  each_color, width=4)
    else:
        draw.rectangle(
            [5, y_indicator - indicator_width // 2,
             5 + indicator_width, y_indicator + indicator_width // 2],
            fill=each_color)
    draw.text([int(indicator_width * 1.5) + 5, y_text], each_color_name,
              (0, 0, 0), font)


def _draw_horizontal_legend(legend_title, legend_items_with_colors,
                            background: tuple = (255, 255, 255),
                            legend_markers: str = "line") -> np.ndarray:
    raise NotImplementedError("Currently not implemented! use vertical legend!")


def _get_font():
    try:
        font = ImageFont.truetype('FreeSans.ttf', 24)
    except IOError:
        font = ImageFont.load_default()
    return font


def _create_random_attributes_colors(
        class_ids_to_names_mapping: Dict[str, Dict[int, str]]
) -> Dict[str, Dict[str, tuple]]:
    colors = {}
    for each_name, each_mapping in class_ids_to_names_mapping.items():
        colors[each_name] = {}
        for each_id, each_class_name in each_mapping.items():
            seed = hash(each_name) + hash(each_id) + hash(each_class_name)
            random.seed(seed)
            random_rgb = tuple(random.randint(0, 255) for _ in range(3))
            colors[each_name][each_id] = random_rgb
    return colors


def _create_random_connections_colors(connection_map: Dict[str, list]
                                      ) -> Dict[str, tuple]:
    colors = {}
    for each_name, each_item in connection_map.items():
        seed = hash(tuple(each_item)) + hash(each_name)
        random.seed(seed)
        random_rgb = tuple(random.randint(0, 255) for _ in range(3))
        colors[each_name] = random_rgb
    return colors
