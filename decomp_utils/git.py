__all__ = [
    "TTY",
    "RE_TEMPLATES",
    "RE_PATTERNS",
    "SotnDecompConsoleFormatter",
    "Spinner",
    "get_repo_root",
    "get_argparser",
    "get_logger",
    "shell",
    "create_table",
    "bar",
    "splat_split",
    "build",
]

def shell(cmd, *, version="us"):
    """Executes a string as a shell command and returns its output"""
    # Todo: Add both list and string handling
    env = os.environ.copy()
    # Ensure the correct VERSION is passed
    env["VERSION"] = version
    cmd_output = run(cmd.split(), env=env, capture_output=True)
    if cmd_output.returncode != 0:
        logger = get_logger()
        logger.warning(cmd_output.stdout)
        logger.error(cmd_output.stderr)
        # raise CalledProcessError(cmd_output.returncode, cmd, cmd_output.stderr)
    return cmd_output.stdout
