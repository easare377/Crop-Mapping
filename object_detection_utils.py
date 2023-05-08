import os
import image_utils as iu
import numpy as np
from PIL import Image, ImageEnhance
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET


def read_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    folder = root.find('folder').text
    filename = root.find('filename').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    depth = int(root.find('size/depth').text)
    # path = root.find('path').text
    list_with_all_boxes = []
    for boxes in root.iter('object'):
        class_name = boxes.find("name").text
        # ymin, xmin, ymax, xmax = None, None, None, None
        xmin = int(boxes.find("bndbox/xmin").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        bbox = (xmin, ymin, xmax, ymax)
        list_with_all_boxes.append((class_name, bbox))
    return folder, filename, (width, height, depth), list_with_all_boxes


def __indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write_xml(xml_path, folder, filename, path, image_dimen, bbox_list):
    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = path
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    # Details from first entry
    # e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = bbox_list[0]
    image_width, image_height, depth = image_dimen
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(image_width)
    SubElement(size, 'height').text = str(image_height)
    SubElement(size, 'depth').text = str(depth)

    SubElement(root, 'segmented').text = '0'

    for bbox_info in bbox_list:
        # e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = entry
        class_name, bbox = bbox_info
        xmin, ymin, xmax, ymax = bbox
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(xmin)
        SubElement(bbox, 'ymin').text = str(ymin)
        SubElement(bbox, 'xmax').text = str(xmax)
        SubElement(bbox, 'ymax').text = str(ymax)

    __indent(root)
    tree = ElementTree(root)
    tree.write(xml_path)


def get_rect(bbox):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    return (xmin, ymin, width, height)


def get_bbox(rect):
    x, y, width, height = rect
    xmin = x
    ymin = y
    xmax = x + width
    ymax = y + height
    return (xmin, ymin, xmax, ymax)


def plot_rect(image, rects, box_color, thickness=1):
    np_image = np.asarray(image)
    np_image = np.copy(np_image)
    for rect in rects:
        x, y, w, h = rect
        r, g, b = box_color
        # get the row to mark.
        # draws top line
        np_image[y: y + thickness, x:x + w] = [r, g, b]
        # draws left line
        np_image[y: y + h, x: x + thickness] = [r, g, b]
        # draws right line
        np_image[y: y + h + thickness, x + w: x + w + thickness] = [r, g, b]
        # draws bottom line
        np_image[y + h: y + h + thickness, x:x + w] = [r, g, b]
    return Image.fromarray(np_image)


def rotate_labeled_image(image, angle, bboxes_info):
    im_w, im_h = image.size
    image = image.rotate(360 - angle)
    new_bboxes = []
    for bbox_info in bboxes_info:
        class_name, bbox = bbox_info
        x, y, w, h = get_rect(bbox)
        if angle == 90:
            bb_x = im_h - (y + h)
            bb_y = x
            bb_w = h
            bb_h = w
            new_bbox = get_bbox((bb_x, bb_y, bb_w, bb_h))
        elif angle == 180:
            bb_x = im_w - (x + w)
            bb_y = im_h - (y + h)
            bb_w = w
            bb_h = h
            new_bbox = get_bbox((bb_x, bb_y, bb_w, bb_h))
        elif angle == 270:
            bb_x = y
            bb_y = im_w - (x + w)
            bb_w = h
            bb_h = w
            new_bbox = get_bbox((bb_x, bb_y, bb_w, bb_h))
        new_bboxes.append((class_name, new_bbox))
    return image, new_bboxes


def flip_labeled_image_left_right(image, bboxes_info):
    im_w, im_h = image.size
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    new_bboxes = []
    for bbox_info in bboxes_info:
        class_name, bbox = bbox_info
        x, y, w, h = get_rect(bbox)
        bb_x = im_w - (x + w)
        bb_y = y
        bb_w = w
        bb_h = h
        new_bbox = get_bbox((bb_x, bb_y, bb_w, bb_h))
        new_bboxes.append((class_name, new_bbox))
    return image, new_bboxes
