import argparse
from github_solutions import *
from scrape_reddit import *



def main():
    parser = argparse.ArgumentParser(description="Process some optional flags.")
    parser.add_argument('--leaderboard', action='store_true', help='Get AdventOfCode leaderboard results')
    parser.add_argument('--reddit', action='store_true', help='Fetch data from r/AdventOfCode subreddit')
    parser.add_argument('--file', type=str, help='Path to .docx file with github repos')

    args = parser.parse_args()

    if args.leaderboard:
        print("AoC leaderboard flag is set.\nScraping AoC leaderboard")
        scrape_aoc_leaderboard()

    if args.reddit:
        print("Reddit flag is set.\nScraping Reddit")
        scrape_reddit()
        # 31523


    if args.file:
        print('File was given. Searching for github repos in file')
        file = args.file
        get_repos_from_file(file)


if __name__ == "__main__":
    main()