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
    conda,
    pypi,
    resources, 
    nvct
)
from metaflow.cards import Markdown

class WhisperTranscriptionFlow(FlowSpec):
    """
    This flow is designed to transcribe Spontaneous Speech datasets
    """
    
    # Sample file 
    sample_file = IncludeFile('sample_file', default='./truth-universally-ack.mp3', is_text=False)
    
    # Define an array of the models
    theModels = ['tiny', 'base', 'small', 'medium', 'large-v3']

    # START FLOW 
    @card(type="default")
    @step
    def start(self):
        """
        Setup
        """
        print ("Starting flow ... ")
        # I know that the Whisper deps step works so skipping to the next one 
        # self.next(self.install_whisper_and_deps)
        self.next(self.get_datasets)

    
    
    
    @conda(python='3.10',
          packages={
            
            })
    @card(type="default")
    @step
    def get_datasets(self):
        """
        Get the datasets
        """
    
        # not sure yet how I am getting the datasets, 
        # possibly through the Python API 
        print("Fetching datasets ... ")
        
        # Documentation for the Python API to Google Cloud Storage
        # https://cloud.google.com/python/docs/reference/storage/latest
        
        # Files are stored at: 
        # https://console.cloud.google.com/storage/browser/_details/common-voice-prod-prod-bundler/sps-corpus-1.0-2025-09-05/
        # with the filename in the format: 
        # sps-corpus-1.0-2025-09-05-[CODE].tar.gz
        # where [CODE] is the language code, e.g. "aat" or "ady"
        
        import os
        import subprocess
        
        # conda doesn't have dotenv
        subprocess.run([
            'pip', 'install', 'dotenv'
        ], check=True)
        
        subprocess.run([
            'pip', 'install', 'google-cloud-storage'
        ], check=True)
        
        
        from dotenv import load_dotenv
        from google.cloud import storage
        
        
        # Load environment variables
        load_dotenv()
        
        # Verify credentials are set
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not creds_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set")
        
        if not os.path.exists(creds_path):
            raise ValueError(f"Credentials file not found: {creds_path}")
        
        print(f"✓ Using credentials from: {creds_path}")
        
        # Test GCS connection
        
        project_id = os.environ.get('GCP_PROJECT_ID')
        client = storage.Client(project=project_id)
        print(f"✓ Connected to GCP project: {project_id}")
        # If we can get to this point I know that the credentials are OK and I can start bringing in the tar files 
        
        # Connected successfully, now I want to pull in the `tar.gz` files somehow
        


        self.next(self.end)

    @step
    def end(self):
        """
        This is the mandatory 'end' step: it prints some helpful information
        to access the model and the used dataset.
        """
        import os 
        
        gcs_bucket_name = os.environ.get('GCS_BUCKET_NAME')
        
        print(
            f"""
            Flow complete.

            See artifacts at {gcs_bucket_name}.
            """
        )


if __name__ == "__main__":
    WhisperTranscriptionFlow()