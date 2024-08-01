import re
from pathlib import Path
from unittest.mock import patch

import pytest

from sgpt.config import cfg
from sgpt.role import DefaultRoles, SystemRole

from .utils import app, cmd_args, comp_args, mock_comp, runner

role = SystemRole.get(DefaultRoles.DEFAULT.value)


@patch("sgpt.handlers.handler.completion")
def test_verbose(completion, monkeypatch: pytest.MonkeyPatch):
    completion.return_value = mock_comp("print('Hello World')")

    prompt = "hello world python"
    args = {"prompt": prompt, "--verbose": True}
    with monkeypatch.context() as m:
        # Disable colors, wrapping and trimming:
        m.setenv("TERM", "dumb")
        m.setenv("COLUMNS", "120")
        m.setenv("LINES", "1000")
        result = runner.invoke(app, cmd_args(**args))

    completion.assert_called_once_with(**comp_args(role, args["prompt"]))
    assert result.exit_code == 0
    assert "print('Hello World')" in result.stdout
    output_lines = result.stdout.splitlines()
    first_line = output_lines[0]
    assert re.fullmatch(r"╭─+ ShellGPT · System Prompt ─+╮╭─+ User ─+╮", first_line)
    left_panel = "\n".join(
        borderless_text
        for line in remove_top_bottom_borders(output_lines)
        if (borderless_text := left_of_border(remove_left_border(line)))
    )
    right_panel = "\n".join(
        borderless_text
        for line in remove_top_bottom_borders(output_lines)
        if (borderless_text := right_of_border(remove_left_right_borders(line)))
    )
    assert left_panel.replace("\n", "") == DefaultRoles.DEFAULT.get_role().role.replace(
        "\n", ""
    )
    assert right_panel.replace("\n", "") == prompt.replace("\n", "")


def remove_top_bottom_borders(output_lines: list[str]) -> list[str]:
    return output_lines[1:-3]


def remove_left_border(line: str) -> str:
    return line[2:]


def remove_left_right_borders(line: str) -> str:
    return line[2:-2]


def left_of_border(line: str) -> str:
    return line.partition(" │")[0].strip()


def right_of_border(line: str) -> str:
    return line.partition("│ ")[2].strip()


@patch("sgpt.printer.TextPrinter.live_print")
@patch("sgpt.printer.MarkdownPrinter.live_print")
@patch("sgpt.handlers.handler.completion")
def test_verbose_no_markdown(completion, markdown_printer, text_printer):
    completion.return_value = mock_comp("print('Hello World')")

    args = {"prompt": "make a commit using git", "--verbose": True, "--md": True}
    result = runner.invoke(app, cmd_args(**args))

    assert result.exit_code == 0
    # Should ignore --md for --code option and output code without markdown.
    markdown_printer.assert_not_called()
    text_printer.assert_called()


@patch("sgpt.handlers.handler.completion")
def test_verbose_stdin(completion):
    completion.return_value = mock_comp("# Hello\nprint('Hello')")

    args = {"prompt": "make comments for code", "--verbose": True}
    stdin = "print('Hello')"
    result = runner.invoke(app, cmd_args(**args), input=stdin)

    expected_prompt = f"{stdin}\n\n{args['prompt']}"
    completion.assert_called_once_with(**comp_args(role, expected_prompt))
    assert result.exit_code == 0
    assert "# Hello" in result.stdout
    assert "print('Hello')" in result.stdout


@patch("sgpt.handlers.handler.completion")
def test_verbose_chat(completion):
    completion.side_effect = [
        mock_comp("print('hello')"),
        mock_comp("print('hello')\nprint('world')"),
    ]
    chat_name = "_test"
    chat_path = Path(cfg.get("CHAT_CACHE_PATH")) / chat_name
    chat_path.unlink(missing_ok=True)

    args = {"prompt": "print hello", "--verbose": True, "--chat": chat_name}
    result = runner.invoke(app, cmd_args(**args))
    assert result.exit_code == 0
    assert "print('hello')" in result.stdout
    assert chat_path.exists()

    args["prompt"] = "also print world"
    result = runner.invoke(app, cmd_args(**args))
    assert result.exit_code == 0
    assert "print('hello')" in result.stdout
    assert "print('world')" in result.stdout

    expected_messages = [
        {"role": "system", "content": role.role},
        {"role": "user", "content": "print hello"},
        {"role": "assistant", "content": "print('hello')"},
        {"role": "user", "content": "also print world"},
        {"role": "assistant", "content": "print('hello')\nprint('world')"},
    ]
    expected_args = comp_args(role, "", messages=expected_messages)
    completion.assert_called_with(**expected_args)
    assert completion.call_count == 2

    args["--shell"] = True
    result = runner.invoke(app, cmd_args(**args))
    assert result.exit_code == 2
    assert "Error" in result.stdout
    chat_path.unlink()
    # TODO: Code chat can be recalled without --code option.


@patch("sgpt.handlers.handler.completion")
def test_verbose_repl(completion):
    completion.side_effect = [
        mock_comp("print('hello')"),
        mock_comp("print('hello')\nprint('world')"),
    ]
    chat_name = "_test"
    chat_path = Path(cfg.get("CHAT_CACHE_PATH")) / chat_name
    chat_path.unlink(missing_ok=True)

    args = {"--repl": chat_name, "--verbose": True}
    inputs = ["__sgpt__eof__", "print hello", "also print world", "exit()"]
    result = runner.invoke(app, cmd_args(**args), input="\n".join(inputs))

    expected_messages = [
        {"role": "system", "content": role.role},
        {"role": "user", "content": "print hello"},
        {"role": "assistant", "content": "print('hello')"},
        {"role": "user", "content": "also print world"},
        {"role": "assistant", "content": "print('hello')\nprint('world')"},
    ]
    expected_args = comp_args(role, "", messages=expected_messages)
    completion.assert_called_with(**expected_args)
    assert completion.call_count == 2

    assert result.exit_code == 0
    assert ">>> print hello" in result.stdout
    assert "print('hello')" in result.stdout
    assert ">>> also print world" in result.stdout
    assert "print('world')" in result.stdout


@patch("sgpt.handlers.handler.completion")
def test_verbose_and_shell(completion):
    args = {"--verbose": True, "--shell": True}
    result = runner.invoke(app, cmd_args(**args))

    completion.assert_not_called()
    assert result.exit_code == 2
    assert "Error" in result.stdout


@patch("sgpt.handlers.handler.completion")
def test_verbose_and_describe_shell(completion):
    args = {"--verbose": True, "--describe-shell": True}
    result = runner.invoke(app, cmd_args(**args))

    completion.assert_not_called()
    assert result.exit_code == 2
    assert "Error" in result.stdout
