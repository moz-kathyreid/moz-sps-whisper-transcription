# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This Metaflow file is intended to: 
# 1. Download the relevant Spontaneous Speech dataset (s)
# TBA: Not sure the best way to hand this - whether it's from GCP or via API and curl from datacollective.mozillafoundation.org.
# 
# 2. Install and configure the Whisper ASR engine
# TBA: Not sure best way to handle this - whether to do it via command line or via a Dockerfile 
# 
# 3. Perform inference on the utterances in the Spontaneous Speech dataset (s)
#  and provide indicators of progress 
# 
# 4. Export inferenced transcriptions to a downloadable file

import os

from metaflow import (
    FlowSpec,
    IncludeFile,
    Parameter,
    card,
    current,
    step,
    environment,
    kubernetes,
)
from metaflow.cards import Markdown

GCS_PROJECT_NAME = "sps-whisper-transcriptions"
GCS_BUCKET_NAME = "sps-whisper-transcriptions-bucket"


class WhisperTranscriptionFlow(FlowSpec):
    """
    This flow is a template for you to use
    for orchestration of your model.
    """

    # Doing a hello world to make sure I can execute remotely
    @card(type="default")
    @step
    def start(self):
        """
        Each flow has a 'start' step.

        You can use it for collecting/preprocessing data or other setup tasks.
        """
        
        print ("starting flow ... ")
        print(f"secret message: {os.getenv('test_var1')}")
        print(f"secret message: {os.getenv('test_var2')}")

        self.next(self.get_datasets)

    @card
    @environment(
        vars={
            "test_var1": "apples",
            "test_var2": "chocolate"
        }
    )

    @step
    def get_datasets(self):
        """
        Get the datasets
        """
        
        # not sure yet how I am getting the datasets, 
        # possibly through the Python API 
        print("Fetching datasets ... ")
        
        self.next(self.end)

    @step
    def end(self):
        """
        This is the mandatory 'end' step: it prints some helpful information
        to access the model and the used dataset.
        """
        print(
            f"""
            Flow complete.

            See artifacts at {GCS_BUCKET_NAME}.
            """
        )


if __name__ == "__main__":
    WhisperTranscriptionFlow()