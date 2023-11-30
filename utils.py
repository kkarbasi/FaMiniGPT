import os


def clone_persian_poems_colab(poet: str='hafez') -> str:
    cmd_str = f"git clone https://github.com/kkarbasi/Persian_poems_corpus.git"
    os.system(cmd_str)
    # read it in to inspect it
    colab_path = f'/content/Persian_poems_corpus/normalized/{poet}_norm.txt'
    with open(colab_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

