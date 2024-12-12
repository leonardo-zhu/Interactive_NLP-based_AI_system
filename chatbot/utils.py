# ANSI escape codes for some common colors
class Colors:
    RESET = '\033[0m'  # Reset the color
    BLUE = '\033[34m'  # Blue text
    GREEN = '\033[32m'  # Green text
    CYAN = '\033[36m'  # Cyan text
    RED = '\033[31m'  # Red text
    YELLOW = '\033[33m'  # Yellow text

def print_colored(message, color, end=''):
    print(color + message + Colors.RESET, end=end)
