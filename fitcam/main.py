import argparse

def parse():
    parser = argparse.ArgumentParser(
        prog='Yoga FitCam'
    )
    parser.add_argument('-i', '--input_file', default='webcam')
    parser.add_argument('-o', '--output_file', default='No output')
    parser.add_argument('-d', '--display', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse()
    print(f'test poetry script')
    print(args)
