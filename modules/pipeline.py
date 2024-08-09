"""
Author: ivana27lita
Date: 5 Agustus 2024
This is the pipeline.py module.
Usage:
- Create TFX Pipeline
"""

from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline


def init_local_pipeline(
    pipeline_root: Text, pipeline_name, metadata_path, components
) -> pipeline.Pipeline:
    """Initiate tfx pipeline

    Args:
        pipeline_root (Text): path to pipeline directory
        pipeline_name (str): pipeline name
        metadata_path (str): path to metadata directory
        components (dict): tfx components

    Returns:
        pipeline.Pipeline: pipeline orchestration
    """
    logging.set_verbosity(logging.INFO)

    beam_args = [
        "--direct_running_mode=multi_processing",
        "----direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path,
        ),
        eam_pipeline_args=beam_args,
    )
