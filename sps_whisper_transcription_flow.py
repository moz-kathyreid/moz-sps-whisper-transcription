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

GCS_PROJECT_NAME = "sps-whisper-transcriptions"
GCS_BUCKET_NAME = "sps-whisper-transcriptions-bucket"

class WhisperTranscriptionFlow(FlowSpec):
    """
    This flow is designed to transcribe Spontaneous Speech datasets
    """
    
    # Sample file 
    sample_file = 'truth-universally-ack.mp3'

    # START FLOW 
    @card(type="default")
    @step
    def start(self):
        """
        Setup
        """
        print ("Starting flow ... ")
        self.next(self.install_whisper_and_deps)

    
    # INSTALL WHISPER AND DEPENDENCIES 
    # Use GPU resources (we check for them below)


    @conda(python='3.10',
          packages={
            'ffmpeg': '',
            'pytorch': '',
            'torchvision': '',
            'torchaudio': '', 
            })
    @nvct # nvidia needed for CUDA # this must go before @step decorator
    @card(type="default")
    @step 
    def install_whisper_and_deps(self): 
        """ 
        Install Whisper and dependencies 
        https://github.com/openai/whisper
        """
        
        # Import statements
        import sys
        import os 
        import subprocess
        
        print ("Starting step to install Whisper and dependencies ... ")
        
    
        # Install using system package manager
        # Note this requires sudo password on the local machine, otherwise it cannot get a PID lock
        # sudo doesn't work
        #print("Installing ffmpeg binary ... ")
        #if os.path.exists('/usr/bin/apt-get'):
            #subprocess.run(['apt-get', 'update'], check=True)
            #subprocess.run(['apt-get', 'install', '-y', 'ffmpeg'], check=True)
        # Add to PATH
        # haven't figured this out yet, may not be needed
        
        # INSTALL WHISPER 
        # CUDA deps should be installed above with @nvct decorator
        
        print ("Whisper: installing Whisper from GitHub ... ")
        # Install whisper from git
        subprocess.run([
            'pip', 'install', 'git+https://github.com/openai/whisper.git'
        ], check=True)
        
        print ("Whisper: checking which whisper and torch packages are installed ... ")
        subprocess.run([
            'pip', 'list', '|', 'grep', 'whisper|torch'
        ], check=True)
        
        
        import whisper
        
        print ("Whisper: checking if whisper package imported ... ")
        
        if 'whisper' not in sys.modules:
            print ('WARNING: openai-whisper package not imported')
        else: 
            print('openai-whisper package successfully imported')
            
        # Check that CUDA has loaded properly and we can access GPUs
        # Mmmm tasty tasty GPUs :D 
        import torch 
        import torchaudio
        import torchvision
        
        print("=== NVIDIA GPU Check ===")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        
        # TODO: Raise proper exceptions with a try except block
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                memory_total = torch.cuda.get_device_properties(i).total_memory
                print(f"  Total memory: {memory_total / 1024**3:.1f} GB")
        else: # stop the flow
            print("ERROR: torch.cuda is not available, exiting.")
            self.next(self.end)
           
        self.next(self.get_datasets)
    
    @card(type="default")
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