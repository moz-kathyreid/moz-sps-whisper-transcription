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
            
        # I am testing this with a test file at the moment 
        # Access the content of the file
        # print(self.sample_file)
        print ("Whisper: transcribing test file ... ")

        import mimetypes
        import tempfile
        import os
        
        print("Sample file information: ")
        
        try:
            # Step 1: Check the IncludeFile object
            print(f"IncludeFile object: {self.sample_file}")
            print(f"IncludeFile type: {type(self.sample_file)}")
            print(f"IncludeFile size: {len(self.sample_file)} bytes")
            
            # Step 2: Temporary file because a binary object doesn't have a file path 
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_file.write(self.sample_file)
                sample_file_path = tmp_file.name
 
            print(f"File path: {sample_file_path}")
    
            # Step 3: Check file existence and basic properties
            print(f"File exists: {os.path.exists(sample_file_path)}")
            if os.path.exists(sample_file_path):
                print(f"File size: {os.path.getsize(sample_file_path)} bytes")
    
            # Step 4: Check first few bytes (this might be where the error occurs)
            print("Reading first few bytes...")
            with open(sample_file_path, 'rb') as f:
                first_bytes = f.read(10)
                print(f"First 10 bytes: {first_bytes}")
                print(f"First 10 bytes hex: {first_bytes.hex()}")
    
            # Step 5: Try mimetypes (this might be where the error occurs)
            print("Checking MIME type...")
            file_type, encoding = mimetypes.guess_type(sample_file_path)
            print(f"MIME type: {file_type}")
            print(f"Encoding: {encoding}")
    
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
        
        transcribe_language = 'en'
        whisper_model = []
        result = []
        
        for index, model in enumerate(self.theModels): 
            print('Loading model: ', model, ' and index is: ', index)
            whisper_model.append(whisper.load_model(model)) 
            
            print('Transcribing using model: ', model)
            result.append(whisper_model[index].transcribe(sample_file_path, language=transcribe_language))
            
            print('Transcription from model ', model, ' is:', result[index])
            
            
            
        # now I want to test writing this information back to a JSON file 
        import json 
        # use artefact so that it persists between @step 
        self.test_transcription = json.dumps(result) 
        # write to a file
        with open('test_transcription.json', 'w') as f:
            f.write(self.test_transcription)
           
        self.next(self.get_datasets)
        
        
        
        
                # Verify credentials are set
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not creds_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set")
        
        if not os.path.exists(creds_path):
            raise ValueError(f"Credentials file not found: {creds_path}")
        
        print(f"✓ Using credentials from: {creds_path}")