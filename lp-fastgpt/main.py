import create_model

import numpy as np


def load_input(filename: str, ) -> tuple[str, int]:
    """Straight transcription of load_input from
    driver.f90."""
    n_tokens_to_generate: int = 20
    input_txt: str = '''Alan Turing theorized that
    computers would one day become very powerful
    , but even he could not imagine'''
    # input_txt2: str = ''
    # u : int = 0
    # ios : int = 0
    return input_txt, n_tokens_to_generate


if __name__ == '__main__':
    print("hello lp_fastgpt")
    create_model.main()
