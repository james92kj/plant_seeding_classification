import pandas as pd
import os

def create_submission(test_paths, predictions, idx_to_label, output_path):

    filenames = [os.path.basename(path) for path in test_paths]
    label_names = [idx_to_label[pred] for pred in predictions]

    df = pd.DataFrame({'file': filenames, 'species': label_names})
    df.to_csv(output_path, index=False)

    print(f'Saved submission to: {output_path}')
