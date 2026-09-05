"""GeoVesselMOT test-set evaluation with the bundled TrackEval subset."""

from multiprocessing import freeze_support

from .TrackEval import trackeval


def eval(dataset, eval_dir, seqmap, exp_name, fps_div, half_eval=False):
    freeze_support()
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5}
    eval_config = {
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": True,
        "LOG_ON_ERROR": None,
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    }

    if not half_eval:
        gt_format = "{gt_folder}/{seq}/gt/gt.txt"
    elif fps_div == 1:
        gt_format = "{gt_folder}/{seq}/gt/gt_val_half.txt"
    else:
        gt_format = f"{{gt_folder}}/{{seq}}/gt/gt_1_{fps_div}.txt"

    dataset_config = {
        "GT_FOLDER": dataset,
        "TRACKERS_FOLDER": eval_dir,
        "OUTPUT_FOLDER": None,
        "TRACKERS_TO_EVAL": [exp_name],
        "CLASSES_TO_EVAL": ["vessel"],
        "BENCHMARK": "GeoVesselMOT",
        "SPLIT_TO_EVAL": "test",
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": False,
        "DO_PREPROC": True,
        "TRACKER_SUB_FOLDER": "",
        "OUTPUT_SUB_FOLDER": "",
        "TRACKER_DISPLAY_NAMES": None,
        "SEQMAP_FOLDER": None,
        "SEQMAP_FILE": seqmap,
        "SEQ_INFO": None,
        "GT_LOC_FORMAT": gt_format,
        "SKIP_SPLIT_FOL": True,
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        metric(metrics_config)
        for metric in (trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity)
    ]
    output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
    summary = output_res["summary"]
    return summary[0]["HOTA"], summary[2]["IDF1"], summary[1]["MOTA"], summary[0]["AssA"]
