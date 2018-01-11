from random import shuffle
import shutil
import os

def shuffle_dataset(ds):
    all_conversations = ds.split('\n\n')
    shuffle(all_conversations)

    return '\n\n'.join(all_conversations)


if __name__=="__main__":
    input_directory = '../data/personalized-dialog-dataset/merged-from-split-by-profile/'
    shuffled_file = 'personalized-dialog-task5-full-dialogs-trn.txt'
    output_directory = '../data/personalized-dialog-dataset/merged-from-split-by-profile-shuffled/'

    shutil.copytree(input_directory, output_directory)

    with open(os.path.join(input_directory, shuffled_file)) as f:
        data = f.read()

    shuffled = shuffle_dataset(data)

    with open(os.path.join(output_directory, shuffled_file), 'w') as f:
        f.write(shuffled)