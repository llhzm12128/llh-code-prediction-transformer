import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parse AST IDs for evaluation")
    parser.add_argument("--ast", help="Filepath with new ASTs")
    parser.add_argument("--out", help="Outfile for ids.txt")
    
    args = parser.parse_args()
    if os.path.exists(args.out):
        os.remove(args.out)
    

if __name__ == "__main__":
    main()