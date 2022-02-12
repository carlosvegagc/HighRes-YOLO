import pandas as pd
import numpy as np
import skimage.io
import shutil
import cv2
import os
import time

import yoltv4.prep_train as prep_train
#import yoltv4.tile_ims_labels as tile_ims_labels
import yoltv4.post_process as post_process

from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

# Function for obtaining the images in a folder


def get_images_dicts(directory):
    dataset_dicts = []
    for filename in [file for file in listdir_nohidden(directory) if not(file.endswith(".csv"))]:
        dataset_dicts.append(filename)

    return dataset_dicts


def rescale(origin_dir, save_dir, grid_pixels_density):

    # Removes the data existing on the save directory.
    if os.path.exists(save_dir) != 0:
        print("Removing existing data")
        shutil.rmtree(save_dir)

    # List data on the origin directory
    images_dir_list = get_images_dicts(origin_dir)

    # Loop for processing each image of the PCB found in the images folder
    for img_filename in images_dir_list:

        # Image name without extension
        img_name = img_filename.split(".")[0]

        # Loads the image
        img = cv2.imread(os.path.join(origin_dir, img_filename))

        # Loads the configuration file
        boardconfigfilename = img_name + "_config.csv"
        if os.path.isfile(os.path.join(origin_dir, boardconfigfilename)):
            pcb_config_df = pd.read_csv(
                os.path.join(origin_dir, boardconfigfilename))

        else:
            raise RuntimeError('Not found config file')

        # Get size of the board
        size_x_mm = pcb_config_df["Board_Width"].iloc[0]
        size_y_mm = pcb_config_df["Board_Height"].iloc[0]
        print("Size of the board is: "+str(size_x_mm) + "x" + str(size_y_mm))

        for index, pixel_density in enumerate(grid_pixels_density):

            # Resize the image to the expected pixel density
            width = int(size_x_mm * pixel_density)
            height = int(size_y_mm * pixel_density)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            # Saves the  Image
            savedir = save_dir + "/"
            if os.path.exists(savedir) == 0:
                os.makedirs(savedir)

            savename = img_name + "_" + str(pixel_density) + ".jpg"
            savename = os.path.join(save_dir, savename)

            cv2.imwrite(savename, img)

    return


def slice_single_image(image_path, out_name, out_dir_images,
                       en_GT=False, boxes=[], classes=[], out_dir_labels=None,
                       mask_path=None, out_dir_masks=None,
                       sliceHeight=416, sliceWidth=416,
                       overlap=0.1, slice_sep='|', pad=0,
                       skip_highly_overlapped_tiles=False,
                       overwrite=False,
                       out_ext='.png', verbose=False):
    """
    Slice a large image into smaller windows, and also bin boxes
    Adapted from:
         https://github.com/avanetten/simrdwn/blob/master/simrdwn/core/slice_im.py

    Arguments
    ---------
    image_path : str
        Location of image to slice
    out_name : str
        Root name of output files (coordinates will be appended to this)
    out_dir_images : str
        Output directory for images
        boxes : arr
                List of bounding boxes in image, in pixel coords
        [ [xb0, yb0, xb1, yb1], ...]
        Defaults to []
    yolo_classes : list
        list of class of objects for each box [0, 1, 0, ...]
        Defaults to []
    out_dir_labels : str
        Output directory for labels
        Defaults to None
    sliceHeight : int
        Height of each slice.  Defaults to ``416``.
    sliceWidth : int
        Width of each slice.  Defaults to ``416``.
    overlap : float
        Fractional overlap of each window (e.g. an overlap of 0.2 for a window
        of size 256 yields an overlap of 51 pixels).
        Default to ``0.1``.
    slice_sep : str
        Character used to separate outname from coordinates in the saved
        windows.  Defaults to ``|``
    out_ext : str
        Extension of saved images.  Defaults to ``.png``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``

    Returns
    -------
    None
    """

    if len(out_ext) == 0:
        im_ext = '.' + image_path.split('.')[-1]
    else:
        im_ext = out_ext

    t0 = time.time()
    # , as_grey=False).astype(np.uint8)  # [::-1]
    image = skimage.io.imread(image_path)
    if verbose:
        print("image.shape:", image.shape)
    if mask_path:
        mask = skimage.io.imread(mask_path)
    win_h, win_w = image.shape[:2]
    #win_size = sliceHeight*sliceWidth
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    n_ims = 0
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):
            out_boxes_yolo = []
            out_classes_yolo = []
            n_ims += 1

            if (n_ims % 20) == 0:
                print(n_ims)

            # make sure we don't have a tiny image on the edge
            if y0+sliceHeight > image.shape[0]:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (y0+sliceHeight - image.shape[0]) > (0.6*sliceHeight):
                        continue
                    else:
                        y = image.shape[0] - sliceHeight
                else:
                    y = image.shape[0] - sliceHeight
            else:
                y = y0
            if x0+sliceWidth > image.shape[1]:
                # skip if too much overlap (> 0.6)
                if skip_highly_overlapped_tiles:
                    if (x0+sliceWidth - image.shape[1]) > (0.6*sliceWidth):
                        continue
                    else:
                        x = image.shape[1] - sliceWidth
                else:
                    x = image.shape[1] - sliceWidth
            else:
                x = x0

            xmin, xmax, ymin, ymax = x, x+sliceWidth, y, y+sliceHeight

            # find boxes that lie entirely within the window
            if en_GT:
                out_path_label = os.path.join(
                    out_dir_labels,
                    out_name + slice_sep + str(y) + '_' + str(x) + '_'
                    + str(sliceHeight) + '_' + str(sliceWidth)
                    + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                    + '.txt')
                for j, b in enumerate(boxes):
                    yolo_class = classes[j]
                    xb0, yb0, xb1, yb1 = b

                    ## CHECKS IF THE CENTER IS INSIDE ##
                    width_box = (xb1 - xb0) * 0.2
                    height_box = (yb1 - yb0) * 0.2

                    if (xb0 + width_box >= xmin) and (yb0 + height_box >= ymin) \
                            and (xb1 - width_box <= xmax) and (yb1 - height_box <= ymax):
                        if(xb0 < xmin):
                            xb0 = xmin + int(sliceWidth * 0.005)
                        if(yb0 < ymin):
                            yb0 = ymin + int(sliceHeight * 0.005)
                        if(xb1 > xmax):
                            xb1 = xmax - int(sliceWidth * 0.005)
                        if(yb1 > ymax):
                            yb1 = ymax - int(sliceHeight * 0.005)

                        # get box coordinates within window
                        out_box_tmp = [xb0 - xmin, xb1 - xmin,
                                       yb0 - ymin, yb1 - ymin]

                        if verbose:
                            print("  out_box_tmp:", out_box_tmp)
                        # out_boxes.append(out_box_tmp)
                        # convert to yolo coords (x,y,w,h)
                        yolo_coords = prep_train.convert((sliceWidth, sliceHeight),
                                                         out_box_tmp)
                        if verbose:
                            print("    yolo_coords:", yolo_coords)
                        out_boxes_yolo.append(yolo_coords)
                        out_classes_yolo.append(yolo_class)

                    #################################

                    # ORIGINAL CODE TO CHECK IF IT IS 100% INside

                    # if (xb0 >= xmin) and (yb0 >= ymin) \
                    #     and (xb1 <= xmax) and (yb1 <= ymax):
                    #     # get box coordinates within window
                    #     out_box_tmp = [xb0 - xmin, xb1 - xmin,
                    #                    yb0 - ymin, yb1 - ymin]

                    #     # out_boxes.append(out_box_tmp)
                    #     # convert to yolo coords (x,y,w,h)
                    #     yolo_coords = prep_train.convert((sliceWidth, sliceHeight),
                    #                            out_box_tmp)
                    #     if verbose:
                    #         print("  out_box_tmp:", out_box_tmp)
                    #         print("    yolo_coords:", yolo_coords)
                    #     out_boxes_yolo.append(yolo_coords)
                    #     out_classes_yolo.append(yolo_class)

                # skip if no labels?
                if len(out_boxes_yolo) == 0:
                    continue

                # save yolo labels
                txt_outfile = open(out_path_label, "w")
                for yolo_class, yolo_coord in zip(out_classes_yolo, out_boxes_yolo):
                    outstring = str(yolo_class) + " " + \
                        " ".join([str(a) for a in yolo_coord]) + '\n'
                    if verbose:
                        print("  outstring:", outstring.strip())
                    txt_outfile.write(outstring)
                txt_outfile.close()

            # save mask, if desired
            if mask_path:
                mask_c = mask[y:y + sliceHeight, x:x + sliceWidth]
                outpath_mask = os.path.join(
                    out_dir_masks,
                    out_name + slice_sep + str(y) + '_' + str(x) + '_'
                    + str(sliceHeight) + '_' + str(sliceWidth)
                    + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                    + im_ext)
                skimage.io.imsave(outpath_mask, mask_c, check_contrast=False)

            # extract image
            window_c = image[y:y + sliceHeight, x:x + sliceWidth]
            outpath = os.path.join(
                out_dir_images,
                out_name + slice_sep + str(y) + '_' + str(x) + '_'
                + str(sliceHeight) + '_' + str(sliceWidth)
                + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                + im_ext)
            if not os.path.exists(outpath):
                skimage.io.imsave(outpath, window_c, check_contrast=False)
            elif overwrite:
                skimage.io.imsave(outpath, window_c, check_contrast=False)
            else:
                print("outpath {} exists, skipping".format(outpath))
    if verbose:
        print("Num slices:", n_ims,
              "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
        print("Time to slice", image_path, time.time()-t0, "seconds")

    return


def slice_images(origin_dir, save_dir,
                 save_dir_GT=None,
                 include_GT=False,
                 sliceHeight=416,
                 sliceWidth=416,
                 slice_overlap=0.1,
                 out_ext=".jpg",
                 verbose=False):

    # SLICE CONFIGURATION #
    #sliceHeight, sliceWidth = 416, 416
    # slice_overlap=0.2
    slice_sep = '__'
    #out_ext = ".jpg"
    overwrite = True
    #verbose = True

    # Create the save folder if it does not exist
    if os.path.exists(save_dir) == 0:
        os.makedirs(save_dir)

    # List data on the origin directory
    im_list = get_images_dicts(origin_dir)

    #################
    # slice images
    for i, im_name in enumerate(im_list):

        im_path = os.path.join(origin_dir, im_name)
        im_tmp = skimage.io.imread(im_path)
        h, w = im_tmp.shape[:2]

        if verbose:
            print(i, "/", len(im_list), im_name, "h, w =", h, w)

        # tile data
        out_name = im_name.split('.')[0]

        if include_GT:
            # Create the save folder if it does not exist
            if os.path.exists(save_dir_GT) == 0:
                os.makedirs(save_dir_GT)

            # Import the dataframe and prepare the data
            df_name = out_name + ".csv"
            df_path = os.path.join(origin_dir, df_name)
            df_ROIs = pd.read_csv(df_path, index_col=False)

            df_Boxes = df_ROIs.drop(columns=["Class"])
            df_Boxes.reset_index(drop=True, inplace=True)
            df_Boxes = df_Boxes.loc[:, ~
                                    df_Boxes.columns.str.contains('^Unnamed')]
            box_array = df_Boxes.to_numpy()

            df_Classes = df_ROIs["Class"]
            df_Classes.reset_index(drop=True, inplace=True)
            class_array = df_Classes.to_numpy()

        else:
            box_array = []
            class_array = []

        slice_single_image(
            im_path, out_name, save_dir,
            sliceHeight=sliceHeight, sliceWidth=sliceWidth,
            overlap=slice_overlap,
            slice_sep=slice_sep,
            skip_highly_overlapped_tiles=False,
            overwrite=overwrite,
            en_GT=include_GT,
            boxes=box_array,
            classes=class_array,
            out_dir_labels=save_dir_GT,
            out_ext=out_ext,
            verbose=verbose)

    #################
    # make list of test files
    im_list_test = []
    for f in sorted([z for z in os.listdir(save_dir) if z.endswith(out_ext)]):
        im_list_test.append(os.path.join(save_dir, f))

    return


def post_process_predictions(df_predictions,
                             raw_im_dir='/wdata',
                             im_ext='.tif',
                             out_dir_root='/root/darknet/results/',
                             detection_thresh=0.2,
                             max_edge_aspect_ratio=2.5,
                             nms_overlap_thresh=0.5,
                             slice_size=416,
                             n_plots=4
                             ):

    ###### CONFIGURATION #########

    test_box_rescale_frac = 1.0
    allow_nested_detections = True
    sep = '__'
    show_labels = False
    out_csv = 'preds_refine.csv'
    out_dir_chips = ''
    chip_rescale_frac = 1.1
    chip_ext = '.png'
    plot_dir = 'preds_plot'
    groupby = 'image_path'
    edge_buffer_test = 1
    verbose = False
    super_verbose = False

    ##############################

    t0 = time.time()

    # a few random variabls that should not need altered
    colors = 40*[(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 140, 255),
                 (0, 255, 125), (125, 125, 125)]

    # Outputs
    outpath_refined_df = os.path.join(out_dir_root, out_csv)
    txt_dir = os.path.join(out_dir_root, 'orig_txt')
    plot_dir = os.path.join(out_dir_root, plot_dir)
    out_dir_chips_tot = os.path.join(out_dir_root, out_dir_chips)
    for d in [txt_dir, plot_dir, out_dir_chips_tot]:
        os.makedirs(d, exist_ok=True)

    ###############
    # convert coords, then make some plots for tiled imagery

    # get image names without appended slice coords
    im_name_root_list = [z.split(sep)[0]
                         for z in df_predictions['im_name'].values]
    df_predictions['im_name_root'] = im_name_root_list

    # filter by prob
    df_predictions = df_predictions[df_predictions['prob'] >= detection_thresh]

    # get image path
    im_path_list = [os.path.join(raw_im_dir, im_name + im_ext) for
                    im_name in df_predictions['im_name_root'].values]
    df_predictions['image_path'] = im_path_list

    # add global coords to dataset
    df_tiled_aug = post_process.augment_df(df_predictions,
                                           testims_dir_tot=raw_im_dir,
                                           slice_sizes=[slice_size],
                                           slice_sep=sep,
                                           edge_buffer_test=edge_buffer_test,
                                           max_edge_aspect_ratio=max_edge_aspect_ratio,
                                           test_box_rescale_frac=test_box_rescale_frac,
                                           rotate_boxes=False)
    # print("df_tiled_aug;", df_tiled_aug)

    # filter out low detections?
    if allow_nested_detections:
        groupby_cat_refine = 'category'
    else:
        groupby_cat_refine = ''
    df_refine = post_process.refine_df(df_tiled_aug,
                                       groupby=groupby,
                                       groupby_cat=groupby_cat_refine,
                                       nms_overlap_thresh=nms_overlap_thresh,
                                       plot_thresh=detection_thresh,
                                       verbose=False)
    if verbose:
        print("df_refine.columns:", df_refine.columns)
        print("df_refine.head:", df_refine.head())
        print("df_refine.iloc[0]:", df_refine.iloc[0])

    # save refined df
    df_refine.to_csv(outpath_refined_df)

    # create color_dict
    color_dict = {}
    for i, c in enumerate(sorted(np.unique(df_refine['category'].values))):
        color_dict[c] = colors[i]

    # create geojsons and plots (make sure to get outputs for all images, even if no predictions)
    print("\nCreating geojsons and plots...")
    im_names_tiled = sorted([z.split('.')[0]
                             for z in os.listdir(raw_im_dir) if z.endswith(im_ext)])
    print(im_names_tiled)
    im_names_set = set(df_refine['im_name_root'].values)
    # im_names_tiled = sorted(np.unique(df_refine['im_name_root']))
    # score_agg_tile = []
    tot_detections = 0
    for i, im_name in enumerate(im_names_tiled):
        if verbose:
            print(i, "/", len(im_names_tiled), im_name)
        im_path = os.path.join(raw_im_dir, im_name + im_ext)
        outfile_plot_image = os.path.join(plot_dir, im_name + '.jpg')

        # if no detections, write empty files
        if im_name not in im_names_set:
            boxes, probs, classes, box_names = [], [], [], []

        # else, get all boxes for this image, create a list of box names too
        else:
            df_filt = df_refine[df_refine['im_name_root'] == im_name]
            boxes = df_filt[['Xmin_Glob', 'Ymin_Glob',
                             'Xmax_Glob', 'Ymax_Glob']].values
            probs = df_filt['prob']
            classes = df_filt['category']
            tot_detections += len(boxes)
            if verbose:
                print(" n boxes:", len(boxes))

            box_names = []
            for j, bbox in enumerate(boxes):
                prob, classs = probs.values[j], classes.values[j]

                box_name_tmp = im_name + '_' + str(classs) + '_' + str(np.round(prob, 3)) + '_' + str(int(bbox[0])) \
                    + '_' + str(int(bbox[1])) + '_' + \
                    str(int(bbox[2])) + '_' + str(int(bbox[3]))
                box_name_tmp = box_name_tmp.replace('.', 'p')
                box_names.append(box_name_tmp)

        label_txt = ''

        # plot
        if i < n_plots:
            print("Making output plot...")
            im_cv2 = cv2.imread(im_path)
            # im_skimage = skimage.io.imread(im_path)
            # im_cv2 = cv2.cvtColor(im_skimage, cv2.COLOR_RGB2BGR)
            post_process.plot_detections(im_cv2, boxes,
                                         gt_bounds=[],
                                         scores=probs,
                                         outfile=outfile_plot_image,
                                         plot_thresh=detection_thresh,
                                         classes=classes,
                                         color_dict=color_dict,
                                         plot_line_thickness=2,
                                         show_labels=show_labels,
                                         alpha_scaling=False, label_alpha_scale=0.85,
                                         compression_level=8,
                                         show_plots=False, skip_empty=False,
                                         test_box_rescale_frac=1,
                                         draw_circle=False, draw_rect=True,
                                         label_txt=label_txt,
                                         verbose=super_verbose, super_verbose=False)

        # extract image chips
        if len(out_dir_chips) > 0:
            image = skimage.io.imread(im_path)
            if verbose:
                print("   Extracting chips around detected objects...")
            for bbox, box_name in zip(boxes, box_names):
                xmin0, ymin0, xmax0, ymax0 = bbox
                # adjust bounding box to be slightly larger
                # rescale output box size if desired, might want to do this
                #    if the training boxes were the wrong size
                if chip_rescale_frac != 1.0:
                    dl = chip_rescale_frac
                    xmid, ymid = np.mean(
                        [xmin0, xmax0]), np.mean([ymin0, ymax0])
                    dx = dl*(xmax0 - xmin0) / 2
                    dy = dl*(ymax0 - ymin0) / 2
                    xmin = max(0, int(np.rint(xmid - dx)))
                    xmax = int(np.rint(xmid + dx))
                    ymin = max(0, int(np.rint(ymid - dy)))
                    ymax = int(np.rint(ymid + dy))
                else:
                    xmin, ymin, xmax, ymax = int(xmin0), int(
                        ymin0), int(xmax0), int(ymax0)
                # print("   box:", box, "xmid", xmid, "ymid", ymid, "newdx2", newdx2, "newdy2", newdy2,
                #             "xmin, xmax, ymin, ymax:", xmin, xmax, ymin, ymax)
                # print("blach:", xmin, ymin, xmax, ymax)
                outpath_chip = os.path.join(
                    out_dir_chips_tot, box_name + chip_ext)
                if not os.path.exists(outpath_chip):
                    # extract image
                    window_c = image[ymin:ymax, xmin:xmax]
                    skimage.io.imsave(outpath_chip, window_c,
                                      check_contrast=False)

    # score_agg_tile = np.array(score_agg_tile)
    # total_score_tile = np.mean(score_agg_tile)
    # print("Total score tile = ", total_score_tile)
    # total_score_native = np.mean(score_agg_native) # / len(im_names_native)
    # print("Total score native = ", total_score_native)
    # print("score_agg_native:", score_agg_native)
    # print("score_agg_tile:", score_agg_tile)
    # total_score = np.mean(np.concatenate((score_agg_native, score_agg_tile)))
    # print("Total score = ", total_score)

    # # mv orig_txt to folder?
    # txt_dir = os.path.join(out_dir_root, 'orig_txt')
    # os.makedirs(txt_dir, exist_ok=True)
    # for (c, pred_txt_path) in pred_files_list:
    #     shutil.move(pred_txt_path, txt_dir)

    print("\nAnalyzed", len(im_names_set),
          "images, detected", tot_detections, "objects")
    print("Exection time = ", time.time() - t0, "seconds")

    return df_refine
