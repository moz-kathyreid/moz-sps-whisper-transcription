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
import subprocess

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

class LocaleInfo:
    """
    A class to encapsulate the information about a locale 
    that is used to represent the language support a locale has from Whisper
    and the locale's closest language 

    Args:
    locale (str): A locale in ISO-639-1 or ISO-639-3 format
    whisper_supported (bool): Whether Whisper supports this locale,
        as determined by determineLanguageSupport()
    closest_language (tuple): A tuple containing (language_code, score)
        where language_code is in ISO-639-1 format and score is between
        1 and 1000 (lower is better)
    """
    
    def __init__(self, locale, whisper_supported, closest_language):   
        self.locale = locale
        self.whisper_supported = whisper_supported
        self.closest_language = closest_language
        
    def __str__(self):
        return f"LocaleInfo(locale='{self.locale}', whisper_supported={self.whisper_supported}', closest_language={self.closest_language})"

class WhisperTranscriptionFlow(FlowSpec):
    """
    This flow is designed to transcribe Spontaneous Speech datasets
    """
    
    # at the moment I haven't looped this to do languages so putting the language here explicitly 
    currentTranscriptionLocale = 'xkl'
    currentTranscriptionLocaleIndex = 56
    
    # Sample file used to prove Whisper deps are working
    sample_file = IncludeFile('sample_file', default='./truth-universally-ack.mp3', is_text=False)

    
    # Define an array of the models
    # I only want transcriptions from large-v3
    #   as the smaller models are generally much less accurate
    # but I have written the code to support the use of additional models in the future if needed
    # theModels = ['tiny', 'base', 'small', 'medium', 'large-v3']
    theModels = ['large-v3']
    
    # Define the paths to the .tar.gz files on Google Cloud Storage 
    sps_bucket = 'common-voice-prod-prod-bundler'
    # followed by the actual language-specific dataset
    # e.g. sps-corpus-1.0-2025-09-05-aat.tar.gz
    sps_version = 'sps-corpus-1.0-2025-09-05'
    sps_filetype = 'tar.gz' # in case it ever changes
    
    # Define all the locales for spontaneous speech 
    # TODO: it would be easier to put these in a txt file and read from them to keep them updated
    spsLocales = [ 'aat', # Arvanitika
                'ady', # Adyghe
                'aln', # Gheg Albanian
                'an', # Aragonese
                'ba', # Bashkir
                'bas', # Basaa
                'bew', # Betawi
                'br', # Breton
                'brx', # Bodo
                'bsy', # Sabah Bisaya
                'bxk', # Bukusu
                'ca', # Catalan
                'cdo', # Eastern Min
                'cgg', # Chiga
                'cpx', # Heng Hua
                'cy', # Welsh
                'de', # German
                'el-CY', # Greek Cypriot
                'fr', # French
                'fy-NL', # Frisian
                'ga-IE', # Irish Gaelic
                'gl', # Galician
                'gsw', # Alsatian
                'gv', # Manx
                'hac', # Gorani
                'hch', # Wixárika 
                'ka', # Georgian
                'kbd', # Kabardian
                'kcn', # Nubi
                'koo', # Konzo
                'kzi', # Kelabit
                'led', # Lendu
                'lij', # Ligurian
                'lke', # Kenyi
                'lth', # Thur
                'lv', # Latvian
                'meh', # Mixteco Yucuhiti 
                'mel', # Melanau
                'mmc', # Michoacán Mazahua
                'ms-MY', # Malaysian
                'msi', # Sabah Malay
                'pne', # Western Penan
                'qxp', # Puno Quechua
                'ru', # Russian
                'ruc', # Ruuli
                'rwm', # Amba
                'sco', # Scots
                'sdo', # Serian Bidayuh
                'seh', # Sena
                'snv', # Sa'ban
                'tob', # Toba Qom
                'top', # Papantla Totonac
                'tr', # Turkish
                'ttj', # Rutoro
                'ukv', # Kuku
                'ush', # Ushojo
                'xkl' #Kenya
    ]
    
    # Define the locales supported by Whisper
    # TODO: it would be easier to put these in a txt file and read from them to keep them updated
    whisperLocales = [
                'en', # English
                'zh', # Chinese 
                'de', # German 
                'es', # Spanish
                'ru', # Russian
                'ko', # Korean 
                'fr', # French 
                'ja', # Japanese
                'pt', # Portuguese
                'tr', # Turkish 
                'pl', # Polish 
                'ca', # Catalan 
                'nl', # Dutch 
                'ar', # Arabic 
                'sv', # Swedish 
                'it', # Italian
                'id', # Indonesian
                'hi', # Hindi 
                'fi', # Finnish 
                'vi', # Vietnamese
                'he', # Hebrew 
                'uk', # Ukrainian 
                'el', # Greek
                'ms', # Malay # Note we have this as ms-MY but I think they're the same 
                'cs', # Czech 
                'ro', # Romanian 
                'da', # Danish 
                'hu', # Hungarian 
                'ta', # Tamil 
                'no', # Norwegian # Not sure if Nynorsk or Bokmal 
                'th', # Thai 
                'ur', # Urdu 
                'hr', # Croatian 
                'bg', # Bulgarian 
                'lt', # Lithuanian 
                'la', # Latin 
                'mi', # Maori 
                'ml', # Malayalam
                'cy', # Welsh 
                'sk', # Slovak 
                'te', # Telugu 
                'fa', # Persian 
                'lv', # Latvian 
                'bn', # Bengali 
                'sr', # Serbian 
                'az', # Azerbaijani 
                'sl', # Slovenian 
                'kn', # Kannada 
                'et', # Estonian 
                'mk', # Macedonian 
                'br', # Breton 
                'eu', # Basque 
                'is', # Icelandic 
                'hy', # Armenian NOTE we use a different code for Armenian too hy-AM
                'ne', # Nepali 
                'mn', # Mongolian 
                'bs', # Bosnian
                'kk', # Kazakh 
                'sq', # Albanian 
                'sw', # Swahili 
                'gl', # Galician 
                'mr', # Marathi 
                'pa', # Punjabi 
                'si', # Sinhala
                'km', # Khmer
                'sn', # Shona 
                'yo', # Yoruba 
                'so', # Somali 
                'af', # Afrikaans 
                'oc', # Occitan 
                'ka', # Georgian 
                'be', # Belarusian 
                'tg', # Tajik 
                'sd', # Sindhi 
                'gu', # Gujarati 
                'am', # Amharic 
                'yi', # Yiddish 
                'lo', # Lao 
                'uz', # Uzbek 
                'fo', # Faroese 
                'ht', # Haitian Creole 
                'ps', # Pashto 
                'tk', # Turkmen 
                'nn', # Nynorsk NOTE this is different to bokmal - we use different language codes
                'mt', # Maltese 
                'sa', # Sanskrit 
                'lb', # Luxembourgish 
                'my', # Myanmar - Burmese 
                'bo', # Tibetan 
                'tl', # Tagalog 
                'mg', # Malagasy 
                'as', # Assamese 
                'tt', # Tatar 
                'haw', # Hawaiian 
                'ln', # Lingala
                'ha', # Hausa 
                'ba', # Bashkir 
                'jw', # Javanese NOTE THIS IS THE WRONG CODE I have raised PR https://github.com/openai/whisper/pull/2669
                'su', # Sundanese 
                'yue', # Cantonese
    ]
    
    def determineLanguageSupport (self, localeList, whisperList): 
        """ 
        Determines the language support in Whisper for a given locale, 
        and if not supported, uses the langcodes library
        to determine the closest 

        Args:
            localeList (list): list of locales in ISO-639-1 or ISO-639-3 format
            whisperList (list): list of locales supported by Whisper given at https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
        Returns: 
            return (object): object containing 'locale', 'whisper_supported', 'closest_language'
        """
        returnArr = [] # an array of dicts 
        
        print('Matching sps locales to Whisper supported locales ...')
        print('There are ', len(localeList), ' locales to match against ', len(whisperList), ' locales supported by Whisper.')
        
        print ("Installing langcodes ... ")
        subprocess.run(['pip', 'install', 'langcodes'], check=True)
        import langcodes

        for spsLocale in localeList: 
            
            # Initialise Dict 
            thisLocale = LocaleInfo(spsLocale, False, None)
            # whisper_supported is False by default and closest_language is None by default 
         
            for whisperLocale in whisperList: 
                # check if there is a match
                if (thisLocale.locale == whisperLocale): 
                    thisLocale.whisper_supported = True 
                    break # don't keep searching
            
            # find the closest language if we don't have support 
            if not thisLocale.whisper_supported: 
                desired = thisLocale.locale
                available = whisperList
                match = langcodes.closest_match(desired, available)
                thisLocale.closest_language = match # this should be a tuple of the language code and the match score
            
            print(thisLocale)
            returnArr.append(thisLocale) 
            
        # return an array of localeInfo objects
        return(returnArr)
    
    def determineTranscriptionLanguage(self, localeInfo): 
        """
        Determines the language(s) for transcription
            
        Args:
            localeInfo (object): an object with the following fields 
            -  locale (str): a locale in ISO-639-1 or ISO-639-3 format
            -  whisper_supported (Boolean): whether Whisper supports this locale, given by determineLanguageSupport() 
            -  closest_language (tuple): the closest language in ISO-639-1 format, and a score between 1 and 1000 determining closeness of match, lower is better
                
        Returns: 
            return (list): a list of one or more locales in ISO-639-1 or ISO-639-3 format, each as a string
        """
            
        # TODO: error checking in case, e.g. an invalid locale is passed
            
        print('now processing: ', localeInfo)
        transcriptionLanguage = [] 
            
        if localeInfo.whisper_supported: 
            # supported by Whisper, use the Whisper locale 
            transcriptionLanguage.append(localeInfo.locale)
        else: 
            if (localeInfo.closest_language[0] == 'und'): # undefined 
                transcriptionLanguage.append('en')
                transcriptionLanguage.append('') # we want to transcribe in both en and _no_ specified language 
            else : # a closest language has been defined 
                transcriptionLanguage.append(localeInfo.closest_language[0])
                # localeInfo.closest_language[1] is how close it is, in case we want to do further processing
                    
        for language in transcriptionLanguage:
            print('transcriptionLanguage for ', localeInfo.locale, ' is: ', language)
            
        return(transcriptionLanguage)  
            
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
        self.next(self.get_locales)

    @conda(python='3.10',
          packages={
            
            })
    @card(type="default")
    @step
    def get_locales (self): 
        """
        Define the locales that will be processed
        
        We need to know if the locale is Whisper-supported
        because if it's not then we need to define which is the closest language
        using the langcodes library 
        https://github.com/rspeer/langcodes?tab=readme-ov-file#comparing-and-matching-languages
        
        To determine if the language is Whisper supported, the file to check is the Tokenizer file in the Whisper source 
        https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
        Note: we are NOT transcribing English en in this run 
        """
        
        self.theLocales = self.determineLanguageSupport(self.spsLocales, self.whisperLocales)
        print(len(self.theLocales), ' locales were processed.')

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
        import io 
        import tarfile 
        import csv
        
        print ("Installing pandas ... ")
        subprocess.run(['pip', 'install', 'pandas'], check=True)
        import pandas as pd
        
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
        
        # Test GCS connection
        project_id = os.environ.get('GCP_PROJECT_ID')
        client = storage.Client(project=project_id)
        print(f"✓ Connected to GCP project: {project_id}")
        # If we can get to this point I know that the credentials are OK and I can start bringing in the tar files 
        
        # Connected successfully, now I want to pull in the `tar.gz` files somehow
        test_locale = self.currentTranscriptionLocale # for testing
        bucket = client.bucket(self.sps_bucket)

        print('bucket is: ', bucket)
        print('bucket name: ', bucket.name)

        path =  self.sps_version + '/' + self.sps_version + '-' + test_locale + '.' + self.sps_filetype
        print('path is: ', path)
        print('full GCS path: gs://{}/{}'.format(bucket.name, path))

        blob = bucket.blob(path)
        print('blob is: ', blob)

        # Now check if it exists
        try:
            exists = blob.exists()
            print(f"File exists: {exists}")
            if exists:
                print(f"Size: {blob.size} bytes")
        except Exception as e:
            print(f"Cannot check existence: {type(e).__name__}: {e}")

        # Download and extract one of the tar files to make sure I can extract it
        # Then I will loop over the files when I have this process nailed down a bit more 
        tar_bytes = blob.download_as_bytes()
        
        # the directory structure for the untarred files is: 
        #    sps-corpus-1.0-2025-09-05-[language_code]
        #      audios/
        #      ss-corpus-[language_code].tsv
        #      ss-reported-audios-[language_code].tsv
    
        
        file_to_extract = 'sps-corpus-1.0-2025-09-05-' + self.currentTranscriptionLocale + '/ss-corpus-' + self.currentTranscriptionLocale + '.tsv'
        
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:
            # Extract a specific file and read it
            print('Untar-ing file ...', file_to_extract)
            testfile = tar.getmember(file_to_extract)
            f = tar.extractfile(testfile)
            self.testtar = f.read().decode('utf-8')
            
            print('Putting the TSV file into a pandas dataframe ...')
            # handle commas in the TSV file by quoting them
            # otherwise you will get the error similar to 
            # pandas.errors.ParserError: Error tokenizing data. C error: Expected 10 fields in line 17, saw 14
            self.df = pd.read_csv(io.StringIO(self.testtar), sep='\t', quoting=csv.QUOTE_ALL, header=0)
            
        print('Saving pandas dataframe ...')
        # save the pandas dataframe to a file and have Metaflow store it as an artefact
        self.test_tsv = self.df.to_csv(sep='\t', index=True) # explicit true row index so I remember to remove if needed
            
        print('Saving tsv to disk ...')
        # also save to disk
        with open('test.tsv', 'w') as f:
            f.write(self.test_tsv)

        self.next(self.do_transcription)
    
    # INSTALL WHISPER AND DEPENDENCIES 
    # Use GPU resources (we check for them below)
    @conda(python='3.10',
          packages={
            'ffmpeg': '',
            'pytorch': '',
            'torchvision': '',
            'torchaudio': '', 
            'tqdm': '', 
            'pandas': ''
            })
    @nvct # nvidia needed for CUDA # this must go before @step decorator
    @card(type="default")
    @step 
    def do_transcription(self): 
        """ 
        Install Whisper and dependencies 
        https://github.com/openai/whisper
        
        Then perform transcription
        """
        
        # Import statements
        import sys
        import os 
        import subprocess
        import io 
        import tarfile 
        import csv
        import traceback
        
        
        subprocess.run([
            'pip', 'install', 'google-cloud-storage'
        ], check=True)

        from google.cloud import storage
        
        print("Starting step to install Whisper and dependencies ... ")
        
        print("Whisper: installing Whisper from GitHub ... ")
        # Install whisper from git
        subprocess.run([
            'pip', 'install', 'git+https://github.com/openai/whisper.git'
        ], check=True)
        
        print("Whisper: checking which whisper and torch packages are installed ... ")
        subprocess.run([
            'pip', 'list', '|', 'grep', 'whisper|torch'
        ], check=True)
        
        
        import whisper
        
        print("Whisper: checking if whisper package imported ... ")
        
        if 'whisper' not in sys.modules:
            print('WARNING: openai-whisper package not imported')
        else: 
            print('openai-whisper package successfully imported')
            
        print("Installing soundfile ... ")
        subprocess.run([
            'pip', 'install', 'soundfile'
        ], check=True)
        
            
        # Check that CUDA has loaded properly and we can access GPUs
        import torch 
        import torchaudio
        import torchvision
        import pandas as pd
        import soundfile as sf
        import numpy as np
        
        print("=== NVIDIA GPU Check ===")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                memory_total = torch.cuda.get_device_properties(i).total_memory
                print(f"  Total memory: {memory_total / 1024**3:.1f} GB")
        else: # stop the flow
            print("ERROR: torch.cuda is not available, exiting.")
            self.next(self.end)
            return
            
        # run a test with the specified locale
        theLocale = self.theLocales[self.currentTranscriptionLocaleIndex]
        print('theLocale is: ', theLocale)
        
        transcribe_language = self.determineTranscriptionLanguage(theLocale)
        print(f"Type: {type(transcribe_language)}")
        print(f"Value: {transcribe_language}")
        print('using transcription language: ', transcribe_language, ' for locale: ', theLocale)
        
        print('there are ', len(transcribe_language), ' languages to transcribe in for this locale ... ')
        
        for lang_index, language in enumerate(transcribe_language): 
            print(f'now transcribing using language: "{language}" which is number {lang_index} of {len(transcribe_language)} in the list ... ')
            
            # copy the dataframe to a new one because we don't iterate over the same dataframe we are modifying
            self.transcription_df = self.df.copy(deep=True)
            
            # let's see the column names of the dataframe and add columns for each of the Whisper models 
            print('the columns in the dataframe are: ')
            print(self.transcription_df.columns)
            print('adding columns for the transcription for each of the models ... ')
            
            for model in self.theModels: 
                column_name = 'transcription_whisper_' + model
                self.transcription_df[column_name] = None # default to None value
                    
            print(self.transcription_df.columns)
            
            # set up the GCS connection because we'll be untar-ing the audio on the fly 
            print('Connecting to GCS to get the compressed audio files ...')
            project_id = os.environ.get('GCP_PROJECT_ID')
            client = storage.Client(project=project_id)
            bucket = client.bucket(self.sps_bucket)
            path = self.sps_version + '/' + self.sps_version + '-' + theLocale.locale + '.' + self.sps_filetype
            print('path is: ', path)
            blob = bucket.blob(path)
                
            self.transcription_verbose_output = [] # to hold the full output from whisper for analysis later
            
            # Download tar file once for this language
            print('Downloading tar file from GCS...')
            tar_bytes = blob.download_as_bytes()
            print(f'Downloaded {len(tar_bytes)} bytes')
            
            # Iterate through the *original* dataframe, but *modify* the new one
            for row_index, row in self.df.iterrows():
                print(f'Processing row: {row_index}')
                    
                try:
                    # find the audio file to transcribe 
                    audio_path = self.sps_version + '-' + theLocale.locale + '/audios/' + self.df.loc[row_index, 'audio_file']
                    print(f'Extracting audio: {audio_path}')
                        
                    # Extract audio file from tar
                    audio_for_transcription = None
                    try:
                        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:
                            print('Untar-ing file ...', audio_path)
                            audio_file = tar.getmember(audio_path)
                            f = tar.extractfile(audio_file)
                            audio_for_transcription = f.read()
                    except KeyError:
                        print(f'ERROR: Audio file not found in tar: {audio_path}')
                        for model in self.theModels:
                            column_name = 'transcription_whisper_' + model
                            self.transcription_df.loc[row_index, column_name] = 'ERROR: Audio file not found in archive'
                        continue
                    except Exception as e:
                        print(f'ERROR extracting audio from tar: {type(e).__name__}: {e}')
                        traceback.print_exc()
                        for model in self.theModels:
                            column_name = 'transcription_whisper_' + model
                            self.transcription_df.loc[row_index, column_name] = f'ERROR: Extraction failed - {str(e)}'
                        continue
                
                    # Determine audio format from file extension
                    audio_format = audio_path.split('.')[-1].lower()
                    print(f'Detected audio format: {audio_format}')
                        
                    # now we perform the transcriptions 
                    print('now performing transcriptions for this audio ... ')
                    for model_index, model in enumerate(self.theModels):
                        try:
                            print(f'Loading model: {model}')
                            whisper_model = whisper.load_model(model)
                                
                            column_name = 'transcription_whisper_' + model
                                
                            # Read audio with explicit format specification
                            audio = None
                            sr = None
                            try:
                                # try not passing audio format 
                                audio, sr = sf.read(io.BytesIO(audio_for_transcription))
                                print(f'Successfully read audio with format: {audio_format}')
                            except Exception as e:
                                print(f'Error reading audio file with format {audio_format}: {e}')
                                # Fallback: try common formats
                                for fmt in ['WAV', 'FLAC', 'OGG', 'MP3', 'OPUS']:
                                    try:
                                        audio, sr = sf.read(io.BytesIO(audio_for_transcription), format=fmt)
                                        print(f'Successfully read with fallback format: {fmt}')
                                        break
                                    except:
                                        continue
                                    else:
                                        raise RuntimeError(f'Could not read audio file with any supported format')
                                
                            # Ensure audio was successfully read
                            if audio is None or sr is None:
                                raise RuntimeError('Audio or sample rate is None after reading')
                                
                            audio = audio.astype(np.float32)
                                
                            print(f'Sample rate: {sr}')
                            print(f'Using model: {model}')
                                
                            # Perform transcription
                            if language:  # if language is not empty string
                                print(f'Transcribing with explicit language: {language}')
                                transcription_output = whisper_model.transcribe(audio, language=language)
                            else:  # language is blank, whisper should choose
                                print('Transcribing with auto-detected language')
                                transcription_output = whisper_model.transcribe(audio)
                                    
                            self.transcription_verbose_output.append({
                                'row_index': row_index,
                                'model': model,
                                'language': language,
                                'output': transcription_output
                            })
                                
                            self.transcription_df.loc[row_index, column_name] = transcription_output['text']
                            print(f'Successfully transcribed row {row_index} with model {model}')
                                
                        except Exception as e:
                            print(f'ERROR transcribing row {row_index} with model {model}: {type(e).__name__}: {e}')
                            traceback.print_exc()
                            self.transcription_df.loc[row_index, column_name] = f'ERROR: {str(e)}'
                            continue
                except Exception as e:
                    print(f'ERROR transcribing row {row_index} with model {model}: {type(e).__name__}: {e}')
                    traceback.print_exc()
                    self.transcription_df.loc[row_index, column_name] = f'ERROR: {str(e)}'
                    continue


        # output to a tsv file 
        output_file = 'whisper_transcriptions_' + theLocale.locale + '_' + (language if language else 'auto') + '.tsv'
        print(f'Saving transcriptions to {output_file}')
        self.transcription_df.to_csv(output_file, sep='\t', index=True)
                
        import json
        verbose_output_file = 'whisper_transcriptions_verbose_output_' + theLocale.locale + '_' + (language if language else 'auto') + '.json'
        print(f'Saving verbose output to {verbose_output_file}')
        with open(verbose_output_file, 'w') as f:
            json.dump(self.transcription_verbose_output, f, indent=2)
                
        print('Transcription step completed')
        self.next(self.end)

    @step
    def end(self):
        """
        This is the mandatory 'end' step: it prints some helpful information
        to access the model and the used dataset.
        """
        print("End flow")


if __name__ == "__main__":
    WhisperTranscriptionFlow()