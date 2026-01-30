from modules.bertopic_mpnet_topics import find_significant_topics


def main() -> None:
    # Example timerange; adjust as needed for your dataset.
    df = find_significant_topics(
        timerange_start="2020-03-04",
        timerange_end="2020-03-13",
        min_topic_count=10,
        min_probability=0.9,
    )
    print(df)


if __name__ == "__main__":
    main()
