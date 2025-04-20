import argparse


def scrape_aoc_leaderboard():



def main():
    parser = argparse.ArgumentParser(description="Process some optional flags.")
    parser.add_argument('--leaderboard', action='store_true', help='Get AdventOfCode leaderboard results')
    parser.add_argument('--reddit', action='store_true', help='Fetch data from r/AdventOfCode subreddit')

    args = parser.parse_args()

    if args.leaderboard:
        print("AoC leaderboard flag is set.\nScraping AoC leaderboard")
        scrape_aoc_leaderboard()

    if args.reddit:
        print("Reddit flag is set.")
        # Add Reddit route

if __name__ == "__main__":
    main()