if __name__ == "__main__":

    import numpy as np

    m = np.memmap('train.bin', dtype=np.uint32, mode='r')
    print(m)
    print(m[-200:])  # print the last 20 tokens to verify
