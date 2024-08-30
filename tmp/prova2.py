import argparse

parser = argparse.ArgumentParser(description="Prova")
parser.add_argument('--no-preproc', dest='preproc', action='store_false', help='Disable data preprocessing')
args = parser.parse_args()
if args.preproc:
    print("Preprocessing attivo")
else:
    print("Preprocessing disattivo")