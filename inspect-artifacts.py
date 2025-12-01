from metaflow import Flow, namespace

# Connect to your Outerbounds setup
namespace(None)

flow_id = 1606
locale = 'bxk'

# Access your flow
flow = Flow('WhisperTranscriptionFlow')
run = flow[flow_id]

# See what artifacts are available
print("Available artifacts:")
for attr in dir(run.data):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Access your specific artifacts
try:
    hasattr(run.data, 'transcription_df')
    print("\nTranscription DataFrame:")
    print(type(run.data.transcription_df))
    print('transcription_df found')
except:
    print("transcription_df not found")

try:
    hasattr(run.data, 'transcription_verbose_output')
    print("\nVerbose output:")
    print(type(run.data.transcription_verbose_output))
    print('verbose output found')
except:
    print("transcription_verbose_output not found")
    
# Save the DataFrame as TSV
run.data.transcription_df.to_csv('whisper_transcription' + '_' + locale + '.tsv', sep='\t', index=True)
print('saved whisper_transcription' + '_' + locale + '.tsv')

# Save the verbose output as JSON
import json
filename = 'whisper_transcription_verbose_output' + '_' + locale + '.tsv'
with open(filename, 'w') as f:
    json.dump(run.data.transcription_verbose_output, f, indent=2)
print('saved ' + filename)