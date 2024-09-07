import logging
import os
import sys
import colorlog

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

# logging.basicConfig(level = logging.INFO,format = '[%(levelname)s] [%(asctime)s]: %(message)s')
color_formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d]: %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
# logging.basicConfig(level=logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(color_formatter)
console_handler.setLevel(logging.DEBUG)
# console_handler.

logger = logging.getLogger("color logger")
for handler in logger.handlers:
    logger.removeHandler(handler)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
