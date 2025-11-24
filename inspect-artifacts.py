from metaflow import Flow, namespace

# Connect to your Outerbounds setup
namespace(None)

flow_id = 1436
locale='aat'

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
    print("\nTranscription DataFrame:")
    print(run.data.transcription_df)
except:
    print("transcription_df not found")

try:
    print("\nVerbose output:")
    print(run.data.transcription_verbose_output)
except:
    print("transcription_verbose_output not found")
    
# Save the DataFrame as TSV
run.data.transcription_df.to_csv('whisper_transcription' + '_' + locale + '.tsv', sep='\t', index=True)

# Save the verbose output as JSON
import json
filename = 'whisper_transcription_verbose_output' + '_' + locale + '.tsv'
with open('local_verbose_output.json', 'w') as f:
    json.dump(run.data.transcription_verbose_output, f, indent=2)