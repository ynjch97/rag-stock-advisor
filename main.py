import argparse

from src.app.stock_advice_workflow import run_stock_advice_workflow


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Stock Advisor MVP 실행 프로그램",
    )
    parser.add_argument(
        "--mode",
        choices=["cli"],
        default="cli",
        help="실행 모드",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()


def run_cli() -> None:
    print("RAG Stock Advisor MVP")
    print("질문을 입력하세요. 빈 값으로 Enter를 누르면 종료합니다.")
    print()

    while True:
        question = input("질문: ").strip()

        if not question:
            print("종료합니다.")
            break

        try:
            answer = run_stock_advice_workflow(question)
        except ValueError as error:
            print(f"오류: {error}")
            print()
            continue

        print()
        print(answer)
        print()


if __name__ == "__main__":
    main()
