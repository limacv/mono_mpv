"""The Code is from LLFF"""

import os
import subprocess
import sys
from .colmap_io_model import *
from .colmap_database import *
from . import colmap_path, is_DEBUG
import sqlite3

colmap_match_type = "exhaustive_matcher"


def printdbg(_st):
    if is_DEBUG:
        print(_st)


def run_colmap(basedir, pipeline=("feature", "match", "mapper"), **kwargs):
    """
    Run_colmap, make sure image is in <basedir>/images/ folder
    pipline: list of str: ["feature", "match", "mapper", "convert"]
    if intrins and extrins is provided, use provided model
    triangulator only happened when provide "camera" and "images" in kwargs
    """
    database_filename = os.path.join(basedir, 'database.db')
    image_path = os.path.join(basedir, 'images')
    tmp_model_path = os.path.join(basedir, "sparse", "tmp")
    out_model_path = os.path.join(basedir, "sparse", "0")

    env_withdll = os.environ.copy()
    dllpath = os.path.dirname(colmap_path)
    if "LD_LIBRARY_PATH" in env_withdll:
        env_withdll["LD_LIBRARY_PATH"] += (':' + dllpath)
    else:
        env_withdll["LD_LIBRARY_PATH"] = dllpath

    if "camera" in kwargs.keys() and "images" in kwargs.keys():
        db = COLMAPDatabase.connect(database_filename)
        db.create_tables()

        thecamera = kwargs["camera"]
        modelid = CAMERA_MODEL_NAMES[thecamera.model].model_id
        cid = db.add_camera(modelid, thecamera.width, thecamera.height, np.array(thecamera.params))
        assert cid == thecamera.id, "COLMAP camera id error when creating database"

        for img_ in kwargs["images"]:
            iid = db.add_image(img_.name, cid)
            assert iid == img_.id, "COLMAP image id error when creating database"
        db.commit()
        db.close()

    if "feature" in pipeline:
        feature_extractor_args = [
            colmap_path, 'feature_extractor',
            '--database_path', database_filename,
            '--image_path', image_path,
            '--ImageReader.camera_model', 'PINHOLE',
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', '0',
        ]
        feat_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True, env=env_withdll))

        printdbg(feat_output)
        printdbg('Features extracted')

    if "match" in pipeline:
        exhaustive_matcher_args = [
            colmap_path, colmap_match_type,
            '--database_path', database_filename,
            "--SiftMatching.use_gpu", '0'
        ]

        match_output = (subprocess.check_output(exhaustive_matcher_args, universal_newlines=True, env=env_withdll))

        printdbg(match_output)
        printdbg('Features matched')

    if "camera" in kwargs.keys() and "images" in kwargs.keys():
        if not os.path.exists(tmp_model_path):
            os.makedirs(tmp_model_path)
        if not os.path.exists(out_model_path):
            os.makedirs(out_model_path)

        write_cameras_text([kwargs["camera"]], os.path.join(tmp_model_path, "cameras.txt"))
        write_images_text(kwargs["images"], os.path.join(tmp_model_path, "images.txt"))
        write_points3D_text((), os.path.join(tmp_model_path, "points3D.txt"))

        if "triangulator" in pipeline:
            triangulator_args = [
                colmap_path, "point_triangulator",
                '--database_path', database_filename,
                '--image_path', image_path,
                '--input_path', tmp_model_path,
                '--output_path', out_model_path,
                '--Mapper.extract_colors', '0',
            ]
            trian_output = (subprocess.check_output(triangulator_args, universal_newlines=True, env=env_withdll))

            printdbg(trian_output)
            printdbg('Triangulartor succeed')

        os.remove(os.path.join(tmp_model_path, "cameras.txt"))
        os.remove(os.path.join(tmp_model_path, "images.txt"))
        os.remove(os.path.join(tmp_model_path, "points3D.txt"))
        os.removedirs(tmp_model_path)

        if "bundle_adjust" in pipeline:
            ba_args = [
                colmap_path, 'bundle_adjuster',
                '--input_path', out_model_path,
                '--output_path', out_model_path,
                '--BundleAdjustment.refine_principal_point', '1',
            ]

            ba_output = (subprocess.check_output(ba_args, universal_newlines=True, env=env_withdll))

            printdbg(ba_output)

    if "mapper" in pipeline:
        p = os.path.join(basedir, 'sparse')
        if not os.path.exists(p):
            os.makedirs(p)

        mapper_args = [
            colmap_path, 'mapper',
            '--database_path', database_filename,
            '--image_path', image_path,
            '--output_path', os.path.join(basedir, 'sparse'),  # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '12',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
        ]

        map_output = (subprocess.check_output(mapper_args, universal_newlines=True, env=env_withdll))

        printdbg(map_output)

    if "convert" in pipeline:
        converter_args = [
            colmap_path, 'model_converter',
            '--input_path', out_model_path,
            '--output_path', out_model_path,
            '--output_type', 'TXT',
        ]

        converter_output = (subprocess.check_output(converter_args, universal_newlines=True, env=env_withdll))

        printdbg('Txt model converted')
        printdbg(converter_output)

    printdbg('Sparse map created')
    if "remove_database" in kwargs.keys() and kwargs["remove_database"]:
        os.remove(database_filename)
