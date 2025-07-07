#!/usr/bin/env python3

"""
Compiler script for the electronics embedding pipeline
"""

import kfp
from electronics_embedding_pipeline import electronics_embedding_pipeline


def compile_pipeline():
    """Compile the pipeline to YAML format for OpenShift AI"""

    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=electronics_embedding_pipeline,
        package_path="electronics_embedding_pipeline.yaml",
    )

    print(" Pipeline compiled successfully!")
    print("   - Output file: electronics_embedding_pipeline.yaml")
    print("   - Ready to upload to OpenShift AI")


if __name__ == "__main__":
    compile_pipeline()
