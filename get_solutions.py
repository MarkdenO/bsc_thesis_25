import argparse
import requests
from collections import defaultdict
from bs4 import BeautifulSoup
from github_solutions import *
import json
import re
import os
import time






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