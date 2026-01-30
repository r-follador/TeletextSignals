from modules.bertopic_mpnet_topics import run_topics


def main() -> None:
    topic_model, summary = run_topics()

    print("BERTopic topics_over_time summary")
    print(f"- documents: {summary['documents']}")
    print(f"- unique topics (excluding -1): {summary['unique_topics']}")
    print(f"- outliers (-1): {summary['outliers']}")
    print(f"- unique teletext_ids: {summary['teletext_ids']}")
    print("\nTop topics (by frequency):")
    topic_info = topic_model.get_topic_info().head(10)
    print(topic_info.to_string(index=False))


if __name__ == "__main__":
    main()
