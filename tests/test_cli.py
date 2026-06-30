from semantic_search_skill.cli import build_parser


def test_query_parser_accepts_core_arguments() -> None:
    parser = build_parser()

    args = parser.parse_args([
        "query",
        "--file-list",
        "tmp/files.txt",
        "--query",
        "local-first AI notes",
        "--top-k",
        "5",
    ])

    assert args.command == "query"
    assert args.file_list == "tmp/files.txt"
    assert args.query == "local-first AI notes"
    assert args.top_k == 5
