import random
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import os
import skimage.io


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}

    for img_file in sorted(os.listdir(img_dir)):
        img = {'object': []}

        img_name = img_file.split(".")[0]

        # Load the current image
        img['filename'] = os.path.join(img_dir, img_file)
        image = skimage.io.imread(img['filename'])
        img['width'], img['height'] = image.shape[:2]

        # Load Annotations file
        ann_path = os.path.join(ann_dir, img_name + ".txt")
        boxes_file = open(ann_path, "r")
        boxes_text = boxes_file.read()

        # Divide the file by lines
        boxes_lines = boxes_text.split("\n")

        for bbox_line in boxes_lines:

            obj = {}

            if len(bbox_line) > 10:
                bbox_elements = bbox_line.split(" ")

                center_x = float(bbox_elements[1])
                center_y = float(bbox_elements[2])
                width = float(bbox_elements[3])
                height = float(bbox_elements[4])

                obj['xmin'] = int(
                    round(img['width'] * (center_x - (width / 2))))
                obj['ymin'] = int(
                    round(img['height'] * (center_y - (height / 2))))
                obj['xmax'] = int(
                    round(img['width'] * (center_x + (width / 2))))
                obj['ymax'] = int(
                    round(img['height'] * (center_y + (height / 2))))

                obj['name'] = bbox_elements[0]

                img['object'] += [obj]

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


def print_anchors(centroids):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '[%0.2f,%0.2f], ' % (anchors[i, 0], anchors[i, 1])

    # there should not be comma after last anchor, that's why
    r += '[%0.2f,%0.2f]' % (anchors[sorted_indices[-1:], 0],
                            anchors[sorted_indices[-1:], 1])
    r += "]"

    print(r)


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity)  # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n, d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum += max(IOU(anns[i], centroids))

    return sum/n


def avg_Distance(centroids):

    # build a complex array of your cells
    z = np.array([complex(c[0], c[1]) for c in centroids])
    # # mesh this array so that you will have all combinations
    # m, n = np.meshgrid(z, z)
    # # get the distance via the norm
    # out = abs(m-n)

    out = abs(z[..., np.newaxis] - z)

    return np.mean(out)


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        # distances.shape = (ann_num, anchor_num)
        distances = np.array(distances)

        # print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def gen_anchors(path_train_imgs, path_train_annot, input_size, labels, num_anchors, num_executions, verbose=False):

    best_avg_iou = 0

    train_imgs, train_labels = parse_annotation(path_train_annot,
                                                path_train_imgs,
                                                labels)
    # run k_mean to find the anchors
    annotation_dims = []
    for image in train_imgs:

        for obj in image['object']:
            width = (float(obj['xmax']) - float(obj['xmin']))  # /cell_w
            height = (float(obj["ymax"]) - float(obj['ymin']))  # /cell_h
            annotation_dims.append(tuple(map(float, (width, height))))

    annotation_dims = np.array(annotation_dims)

    for i in range(num_executions):

        centroids = run_kmeans(annotation_dims, num_anchors)

        avg_iou = avg_IOU(annotation_dims, centroids)

        if best_avg_iou < avg_iou:

            best_avg_iou = avg_iou
            best_centroids = centroids

            avg_dist_best_centroids = avg_Distance(best_centroids)

            if verbose:
                print("Best results on ", i, " score: ", '%0.3f' % avg_iou)
                print("Distance : {}".format(avg_dist_best_centroids))
                print_anchors(centroids)

        if best_avg_iou == avg_iou:

            avg_dist_centroids = avg_Distance(centroids)

            if avg_dist_centroids > avg_dist_best_centroids:
                avg_dist_best_centroids = avg_dist_centroids
                best_avg_iou = avg_iou
                best_centroids = centroids

                if verbose:
                    print("Best results on ", i, " score: ", '%0.3f' % avg_iou)
                    print("Distance : %d".format(avg_dist_best_centroids))
                    print_anchors(centroids)

    # write anchors to file
    #print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_iou)
    # print_anchors(centroids)

    return best_centroids, annotation_dims, best_avg_iou
